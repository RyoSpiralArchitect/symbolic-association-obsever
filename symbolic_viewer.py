import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def choose_device():
    """
    利用可能なデバイスを選ぶ:
    1. MPS (Apple Silicon)
    2. CUDA
    3. CPU
    """
    if torch.backends.mps.is_available():
        print("[INFO] Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[INFO] Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("[INFO] Using CPU")
        return torch.device("cpu")


class SymbolicViewer:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        symbolic_words=None,
        max_new_tokens: int = 80,
        temperature: float = 1.0,
        top_p: float = 0.9,
        capture_hidden: bool = False,
        capture_attn: bool = False,
        save_states_dir: str = "",
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.capture_hidden = capture_hidden
        self.capture_attn = capture_attn
        self.save_states_dir = save_states_dir or ""
        if self.save_states_dir:
            os.makedirs(self.save_states_dir, exist_ok=True)

        # 象徴語のリスト（小文字で保持）
        if symbolic_words is None:
            symbolic_words = [
                "death",
                "blood",
                "moon",
                "dream",
                "night",
                "shadow",
                "ghost",
                "angel",
                "demon",
                "void",
                "grave",
                "bone",
                "skull",
                "silence",
            ]
        self.symbolic_words = {w.lower() for w in symbolic_words}

        print(f"[INFO] モデル読み込み中: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] モデルを {self.device} に配置しました。")

        # GPT-2 は pad_token が未定義なことがあるので EOS と揃える
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sample_id = 0  # アノテーションごとに増やす

    # ===== 基本処理 =====

    def _encode(self, text: str):
        """テキストをエンコードして指定デバイスに送る"""
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def generate_continuation(self, prompt: str):
        """
        プロンプトから続きのテキストをサンプル生成し、
        必要なら full シーケンスの hidden / attention も取得する。
        """
        inputs = self._encode(prompt)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_ids = output_ids[0]  # [seq_len]
        gen_ids = full_ids[input_len:]  # 生成部分だけ

        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        continuation = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # ===== hidden / attention 取得 (任意) =====
        hidden_states = None
        attentions = None

        if self.capture_hidden or self.capture_attn:
            with torch.no_grad():
                out = self.model(
                    full_ids.unsqueeze(0).to(self.device),
                    output_hidden_states=self.capture_hidden,
                    output_attentions=self.capture_attn,
                )
            if self.capture_hidden:
                # list[layer] of [seq, dim]
                hidden_states = [h[0].cpu() for h in out.hidden_states]
            if self.capture_attn:
                # list[layer] of [heads, seq, seq]
                attentions = [a[0].cpu() for a in out.attentions]

        return {
            "full_ids": full_ids.cpu(),
            "gen_ids": gen_ids.cpu(),
            "input_len": input_len,
            "full_text": full_text,
            "continuation": continuation,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }

    def tokenize_and_mark_symbolic_from_ids(self, gen_ids: torch.Tensor):
        """
        生成部分の token IDs から象徴語をマーキングする。
        戻り値: token_infos (list[dict])
        """
        ids = gen_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        token_infos = []

        print("=== Token analysis (generated part) ===")
        for i, (tok, tid) in enumerate(zip(tokens, ids)):
            piece = self.tokenizer.decode([tid])
            norm = piece.strip().lower()
            is_sym = norm in self.symbolic_words

            mark = " <=== SYMBOL" if is_sym else ""
            print(
                f"{i:3d}: tok={tok!r:20s} "
                f"id={tid:6d} dec={piece!r}{mark}"
            )

            token_infos.append(
                {
                    "index": i,          # 生成部分内の index
                    "id": tid,
                    "token": tok,
                    "decoded": piece,
                    "normalized": norm,
                    "is_symbolic_seed": is_sym,
                }
            )

        print()
        return token_infos

    # ===== アノテーション周り =====

    def interactive_annotation(
        self,
        prompt: str,
        gen_text: str,
        token_infos,
        annotations_path: str,
        input_len: int,
        hidden_states,
        attentions,
    ):
        """
        CLI 上で簡易アノテーション & 必要なら hidden / attn を保存。
        - 象徴的だと思うトークンの index をカンマ区切りで入力
        - タグとメモを聞いて JSONL に保存
        - hidden/attn が有効なら、そのトークン位置の状態を .pt 保存
        """
        print("アノテーションしたいトークンの index をカンマ区切りで入力 (例: 5,12,19)")
        print("Enter でスキップできます。")
        idx_line = input("indices > ").strip()
        if idx_line == "":
            print("[INFO] アノテーションはスキップされました。\n")
            return

        try:
            indices = [
                int(x.strip())
                for x in idx_line.split(",")
                if x.strip() != ""
            ]
        except ValueError:
            print("[WARN] index のパースに失敗しました。スキップします。\n")
            return

        records = []
        for tok_idx in indices:
            if tok_idx < 0 or tok_idx >= len(token_infos):
                print(f"[WARN] index {tok_idx} は範囲外なのでスキップします。")
                continue

            info = token_infos[tok_idx]
            seq_pos = input_len + tok_idx  # full シーケンス内の位置

            print(
                f"\nToken {tok_idx}: dec={info['decoded']!r} "
                f"(normalized={info['normalized']!r}, seq_pos={seq_pos})"
            )
            tags_line = input(
                "タグ (例: mythic,phonetic,personal) > "
            ).strip()
            note = input("メモ (自由記述) > ").strip()

            tags = [
                t.strip()
                for t in tags_line.split(",")
                if t.strip() != ""
            ]

            state_file = None
            if self.save_states_dir and (hidden_states is not None or attentions is not None):
                # このトークン用の hidden / attn を切り出して保存
                state_payload = {}

                if hidden_states is not None:
                    # list[layer] of [seq, dim] -> list[layer] of [dim]
                    state_payload["hidden_states"] = [
                        h[seq_pos].clone() for h in hidden_states
                    ]

                if attentions is not None:
                    # list[layer] of [heads, seq, seq] -> list[layer] of [heads, seq]
                    state_payload["attentions_from_token"] = [
                        a[:, seq_pos, :].clone() for a in attentions
                    ]

                state_file = os.path.join(
                    self.save_states_dir,
                    f"sample{self.sample_id}_tok{tok_idx}.pt",
                )
                try:
                    torch.save(state_payload, state_file)
                    print(f"[INFO] hidden/attn を {state_file} に保存しました。")
                except Exception as e:
                    print(f"[WARN] hidden/attn 保存に失敗: {e}")
                    state_file = None

            record = {
                "sample_id": self.sample_id,
                "prompt": prompt,
                "generated_text": gen_text,
                "token_index_in_generation": tok_idx,
                "seq_pos_in_full": seq_pos,
                "token_id": info["id"],
                "token": info["token"],
                "decoded": info["decoded"],
                "normalized": info["normalized"],
                "is_symbolic_seed": info["is_symbolic_seed"],
                "tags": tags,
                "note": note,
                "state_file": state_file,
            }
            records.append(record)

        if not records:
            print("[INFO] 有効なアノテーションがありませんでした。\n")
            return

        # JSONL に追記
        try:
            with open(annotations_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[INFO] {len(records)} 件のアノテーションを {annotations_path} に保存しました。\n")
        except Exception as e:
            print(f"[ERROR] アノテーション保存中にエラー: {e}\n")

        self.sample_id += 1  # 次のサンプルへ


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpiralReality: 生の出力 → 象徴語検出 → hidden/attn 抽出 → アノテーション (MPS 対応)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="ローカルの GPT-2 などのモデルディレクトリ or HF のリポジトリ名 (例: gpt2)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="生成するトークン数 (default: 80)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="サンプリング温度 (default: 1.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help=" nucleus sampling の top_p (default: 0.9)",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="annotations.jsonl",
        help="アノテーションを書き込む JSONL ファイルパス (default: annotations.jsonl)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="象徴語候補 (カンマ区切り, 例: death,moon,blood)。空ならデフォルトリストを使用。",
    )
    parser.add_argument(
        "--capture_hidden",
        action="store_true",
        help="hidden states を取得して保存可能にする",
    )
    parser.add_argument(
        "--capture_attn",
        action="store_true",
        help="attention weights を取得して保存可能にする",
    )
    parser.add_argument(
        "--save_states_dir",
        type=str,
        default="",
        help="hidden/attn を .pt として保存するディレクトリ。空なら保存しない。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device()

    # 象徴語の上書きがあれば反映
    if args.symbols.strip():
        sym_list = [
            w.strip()
            for w in args.symbols.split(",")
            if w.strip() != ""
        ]
    else:
        sym_list = None

    viewer = SymbolicViewer(
        args.model_path,
        device=device,
        symbolic_words=sym_list,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        capture_hidden=args.capture_hidden,
        capture_attn=args.capture_attn,
        save_states_dir=args.save_states_dir,
    )

    print("==============================================")
    print(" SpiralReality Step: 生出力 → 象徴語トークン検出 → hidden/attn 抽出 → アノテーション")
    print(" 空行のみで Enter を押すと終了します。")
    print("==============================================\n")

    while True:
        try:
            prompt = input("input the context > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 終了します。")
            break

        if prompt == "":
            print("[INFO] 終了します。")
            break

        # 1) 生成 + hidden/attn 抽出
        gen = viewer.generate_continuation(prompt)
        full_text = gen["full_text"]
        continuation = gen["continuation"]

        print("\n=== Prompt ===")
        print(prompt)
        print("\n=== Generated continuation ===")
        print(continuation)
        print()

        # 2) 生成部分だけを token 分解して象徴語マーク
        token_infos = viewer.tokenize_and_mark_symbolic_from_ids(gen["gen_ids"])

        # 3) その場でアノテーション (任意、hidden/attn 付き)
        viewer.interactive_annotation(
            prompt=prompt,
            gen_text=continuation,
            token_infos=token_infos,
            annotations_path=args.annotations_path,
            input_len=gen["input_len"],
            hidden_states=gen["hidden_states"],
            attentions=gen["attentions"],
        )

        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
