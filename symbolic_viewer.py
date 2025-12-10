"""
SpiralReality Symbolic Trace Captor (STC)

生の生成過程で現れた象徴語トークンにフォーカスし、
その瞬間の hidden/attention を切り出して象徴 ↔ 潜在の対応を作る装置。

v1.1: token レベルに加えて、文・段落っぽいチャンク（segment）レベルの
      アノテーションもできるように拡張。
"""

import argparse
import json
import os
import re
import hashlib
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
        model_file: str = "",
        offline: bool = False,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.capture_hidden = capture_hidden
        self.capture_attn = capture_attn
        self.save_states_dir = save_states_dir or ""
        self.offline = offline
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
        self.symbol_pattern = re.compile(
            "|".join(re.escape(w) for w in self.symbolic_words), re.IGNORECASE
        )

        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("[INFO] HF_HUB_OFFLINE=1 でローカルファイルのみを使用します。")

        state_dict = None
        if model_file:
            print(f"[INFO] ローカル state_dict を読み込み: {model_file}")
            state_dict = torch.load(model_file, map_location="cpu")

        print(f"[INFO] モデル読み込み中: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=self.offline
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            state_dict=state_dict,
            local_files_only=self.offline,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] モデルを {self.device} に配置しました。")

        # GPT-2 は pad_token が未定義なことがあるので EOS と揃える
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # サンプルID（1回の生成につき +1）
        self.sample_id = 0

    # ===== 基本処理 =====

    def _encode(self, text: str):
        """テキストをエンコードして指定デバイスに送る"""
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    def _short_hash_text(self, text: str) -> int:
        """prompt + continuation を安定ハッシュして 10 桁に短縮"""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest, 16) % 10**10

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
                if out.attentions is None:
                    print(
                        "[WARN] モデルが attention を返しませんでした。"
                        " output_attentions をサポートしていない可能性があります。"
                    )
                    attentions = None
                else:
                    attn_list = []
                    for layer_idx, a in enumerate(out.attentions):
                        if a is None:
                            print(
                                f"[WARN] attention 出力 (layer {layer_idx}) が None でした。"
                                " 該当層をスキップします。"
                            )
                            continue
                        attn_list.append(a[0].cpu())

                    if not attn_list:
                        print("[WARN] 全ての層で attention を取得できませんでした。")
                        attentions = None
                    else:
                        if len(attn_list) != len(out.attentions):
                            print("[WARN] 一部の層でのみ attention を取得しました。")
                        # list[layer] of [heads, seq, seq]
                        attentions = attn_list

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
        同時に、簡易な文/段落セグメントIDも付与する。
        戻り値: token_infos (list[dict]), segments (dict[segment_id] -> [token_index])
        """
        ids = gen_ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        token_infos = []
        segments = {}
        segment_id = 0

        print("=== Token analysis (generated part) ===")
        for i, (tok, tid) in enumerate(zip(tokens, ids)):
            piece = self.tokenizer.decode([tid])
            norm = piece.strip().lower()
            is_sym = bool(self.symbol_pattern.search(piece))

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
                    "segment_id": segment_id,
                }
            )
            segments.setdefault(segment_id, []).append(i)

            # セグメント境界のヒューリスティック：
            # 改行や文末（. ! ?）で次の segment へ
            if "\n" in piece:
                segment_id += 1
            elif any(punct in piece for punct in [".", "!", "?"]):
                segment_id += 1

        print()
        return token_infos, segments

    # ===== token アノテーション =====

    def interactive_annotation(
        self,
        sample_id: int,
        prompt: str,
        gen_text: str,
        token_infos,
        annotations_path: str,
        input_len: int,
        hidden_states,
        attentions,
    ):
        """
        CLI 上で簡易アノテーション & 必要なら hidden / attn を保存（token単位）。
        - 象徴的だと思うトークンの index をカンマ区切りで入力
        """
        print("トークン単位でアノテーションしたい index をカンマ区切りで入力 (例: 5,12,19)")
        print("Enter でスキップできます。")
        idx_line = input("token indices > ").strip()
        if idx_line == "":
            print("[INFO] token アノテーションはスキップされました。\n")
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
        hash_txt = self._short_hash_text(prompt + gen_text)

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
                state_dir = os.path.join(
                    self.save_states_dir, f"sample{sample_id}"
                )
                os.makedirs(state_dir, exist_ok=True)

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
                    state_dir,
                    f"tok{tok_idx}_{hash_txt}.pt",
                )
                try:
                    torch.save(state_payload, state_file)
                    print(f"[INFO] hidden/attn を {state_file} に保存しました。")
                except Exception as e:
                    print(f"[WARN] hidden/attn 保存に失敗: {e}")
                    state_file = None

            record = {
                "level": "token",
                "sample_id": sample_id,
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
                "model_meta": {
                    "num_layers": len(hidden_states) if hidden_states is not None else None,
                    "hidden_dim": hidden_states[0].shape[-1]
                    if hidden_states is not None
                    else None,
                    "num_heads": attentions[0].shape[0] if attentions else None,
                },
            }
            records.append(record)

        if not records:
            print("[INFO] 有効な token アノテーションがありませんでした。\n")
            return

        # JSONL に追記
        try:
            with open(annotations_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[INFO] {len(records)} 件の token アノテーションを {annotations_path} に保存しました。\n")
        except Exception as e:
            print(f"[ERROR] アノテーション保存中にエラー: {e}\n")

    # ===== segment アノテーション =====

    def interactive_span_annotation(
        self,
        sample_id: int,
        prompt: str,
        gen_text: str,
        token_infos,
        segments,
        annotations_path: str,
        input_len: int,
        hidden_states,
        attentions,
    ):
        """
        文・段落っぽいチャンク（segment）単位のアノテーション。
        - segment_id ごとにテキストを表示
        - 選ばれた segment について、属するトークンの hidden/attn を平均して保存
        """
        if not segments:
            print("[INFO] セグメントがありません（空の生成？）。\n")
            return

        print("=== Segments (sentence/paragraph-ish) ===")
        for seg_id, idxs in segments.items():
            seg_text = "".join(token_infos[i]["decoded"] for i in idxs)
            oneline = seg_text.replace("\n", "\\n")
            if len(oneline) > 120:
                oneline = oneline[:117] + "..."
            print(f"[{seg_id}] {oneline}")
        print()

        print("セグメント単位でアノテーションしたい ID をカンマ区切りで入力 (例: 0,2)")
        print("Enter でスキップできます。")
        idx_line = input("segment ids > ").strip()
        if idx_line == "":
            print("[INFO] segment アノテーションはスキップされました。\n")
            return

        try:
            seg_ids = [
                int(x.strip())
                for x in idx_line.split(",")
                if x.strip() != ""
            ]
        except ValueError:
            print("[WARN] segment ID のパースに失敗しました。スキップします。\n")
            return

        records = []
        hash_txt = self._short_hash_text(prompt + gen_text)

        for seg_id in seg_ids:
            if seg_id not in segments:
                print(f"[WARN] segment {seg_id} は存在しません。スキップします。")
                continue

            idxs = segments[seg_id]
            seg_text = "".join(token_infos[i]["decoded"] for i in idxs)

            print(f"\nSegment {seg_id}:")
            print(seg_text)
            tags_line = input(
                "タグ (例: mythic,trauma,relationship) > "
            ).strip()
            note = input("メモ (このセグメント全体の意味・象徴的雰囲気など) > ").strip()

            tags = [
                t.strip()
                for t in tags_line.split(",")
                if t.strip() != ""
            ]

            state_file = None
            if self.save_states_dir and (hidden_states is not None or attentions is not None):
                state_payload = {}
                state_dir = os.path.join(
                    self.save_states_dir, f"sample{sample_id}"
                )
                os.makedirs(state_dir, exist_ok=True)

                # セグメントに属する full シーケンスポジションを取得
                positions = [
                    input_len + token_infos[i]["index"] for i in idxs
                ]

                if hidden_states is not None:
                    # list[layer] of [seq, dim] -> list[layer] of [dim] (セグメント平均)
                    state_payload["hidden_states_segment_mean"] = [
                        h[positions].mean(dim=0).clone() for h in hidden_states
                    ]

                if attentions is not None:
                    # list[layer] of [heads, seq, seq] -> list[layer] of [heads, seq] (from segment 平均)
                    state_payload["attentions_from_segment_mean"] = [
                        a[:, positions, :].mean(dim=1).clone() for a in attentions
                    ]

                state_file = os.path.join(
                    state_dir,
                    f"segment{seg_id}_{hash_txt}.pt",
                )
                try:
                    torch.save(state_payload, state_file)
                    print(f"[INFO] segment hidden/attn を {state_file} に保存しました。")
                except Exception as e:
                    print(f"[WARN] segment hidden/attn 保存に失敗: {e}")
                    state_file = None

            record = {
                "level": "segment",
                "sample_id": sample_id,
                "segment_id": seg_id,
                "segment_token_indices": segments[seg_id],
                "segment_text": seg_text,
                "prompt": prompt,
                "generated_text": gen_text,
                "tags": tags,
                "note": note,
                "state_file": state_file,
                "model_meta": {
                    "num_layers": len(hidden_states) if hidden_states is not None else None,
                    "hidden_dim": hidden_states[0].shape[-1]
                    if hidden_states is not None
                    else None,
                    "num_heads": attentions[0].shape[0] if attentions else None,
                },
            }
            records.append(record)

        if not records:
            print("[INFO] 有効な segment アノテーションがありませんでした。\n")
            return

        try:
            with open(annotations_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[INFO] {len(records)} 件の segment アノテーションを {annotations_path} に保存しました。\n")
        except Exception as e:
            print(f"[ERROR] segment アノテーション保存中にエラー: {e}\n")


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
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="ローカルに保存した model state_dict (.pth) のパス。指定すると from_pretrained の state_dict に適用。",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="HF Hub にアクセスせず、ローカルファイルのみで読み込む (HF_HUB_OFFLINE=1)。",
    )
    parser.add_argument(
        "--span_annotation",
        action="store_true",
        help="文・段落っぽいセグメント単位でもアノテーションする。",
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
        model_file=args.model_file,
        offline=args.offline,
    )

    print("==============================================")
    print(" SpiralReality STC: 生出力 → 象徴語トークン検出 → hidden/attn 抽出")
    print(" token / segment アノテーション対応")
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

        sample_id = viewer.sample_id

        # 1) 生成 + hidden/attn 抽出
        gen = viewer.generate_continuation(prompt)
        continuation = gen["continuation"]

        print("\n=== Prompt ===")
        print(prompt)
        print("\n=== Generated continuation ===")
        print(continuation)
        print()

        # 2) 生成部分だけを token 分解して象徴語マーク & segment 分割
        token_infos, segments = viewer.tokenize_and_mark_symbolic_from_ids(gen["gen_ids"])

        # 3) token アノテーション (任意)
        viewer.interactive_annotation(
            sample_id=sample_id,
            prompt=prompt,
            gen_text=continuation,
            token_infos=token_infos,
            annotations_path=args.annotations_path,
            input_len=gen["input_len"],
            hidden_states=gen["hidden_states"],
            attentions=gen["attentions"],
        )

        # 4) segment アノテーション (任意、フラグが立っているとき)
        if args.span_annotation:
            viewer.interactive_span_annotation(
                sample_id=sample_id,
                prompt=prompt,
                gen_text=continuation,
                token_infos=token_infos,
                segments=segments,
                annotations_path=args.annotations_path,
                input_len=gen["input_len"],
                hidden_states=gen["hidden_states"],
                attentions=gen["attentions"],
            )

        viewer.sample_id += 1  # 次のサンプルへ
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
