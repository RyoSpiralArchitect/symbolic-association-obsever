# Symbolic Association Observer

同じ `moon` が、天体・比喩・神話でどう使われるか。人が付けた注釈と、保存済みのモデル表現を横に並べる小さなローカル実験室です。

## まず触る（Python 3.10+、追加依存なし）

```sh
python trace_inspector.py --demo
# ブラウザで http://127.0.0.1:8765 を開く
```

3つの **手書きテキスト** を検索・比較できます。モデル、API、GPU、ネット接続は不要です。デモには token ID や tensor はなく、数値を捏造しません。終了は Ctrl+C。ポートは `--port 8766` で変更できます。

## 自分の記録を開く

```sh
python trace_inspector.py --annotations /path/to/annotations.jsonl
```

既存 `symbolic_viewer.py` の token / segment JSONL をそのまま読みます。タグ、メモ、プロンプト、生成文、保存された位置を表示し、語またはセグメントで絞り込みます。文章中の下線は最初に一致する文字列の目印で、token alignment の復元ではありません。

`.pt` も調べる場合は **自分で採取した信頼できるファイルのみ** を指定し、採取に使った PyTorch 環境で起動します。

```sh
python trace_inspector.py \
  --annotations /path/to/capture/annotations.jsonl \
  --states-root /path/to/capture
```

`--states-root` は **採取時の作業ディレクトリ** です。例えば JSONL の `state_file` が `states/sample0/tok3_123.pt` なら、その `states/` の親を指定します。絶対パスもこの root 内だけを許可します。移動済みの古い絶対パスは JSONL のコピー側で調整してください。シンボリックリンクによる root 外参照も拒否します。

- 対応キー: `hidden_states` / `hidden_states_segment_mean`、`attentions_from_token` / `attentions_from_segment_mean`
- hidden は保存 index ごとの norm、2記録間の cosine similarity と L2 distance。ゼロベクトルの cosine は未定義として表示
- attention は head 平均の上位5つの全文位置。元 token 列がないため語に変換しません。保存時に欠けた層が詰められた場合を考慮し、元の層番号とは呼びません
- 欠落した state、attention のみ、PyTorch 未導入、壊れたファイルは注釈閲覧を妨げず表示
- token と segment 平均、hidden の層数・次元が違う記録は比較不可
- 旧形式はモデル識別子・tokenizer・採取設定を保存しません。同じモデル・tokenizer・設定かを自分で確認してからチェックを入れてください。形状一致は互換性の証明ではありません
- 読み取り専用、127.0.0.1 に限定。ファイルアップロード・外部API・自動ダウンロードなし。`torch.load(..., weights_only=True, map_location="cpu")` を使い、旧 PyTorch 向けに危険な pickle 読込へフォールバックしません。weights-only も悪意あるファイルやリソース枯渇に対する完全な隔離ではありません
- 小さな実験向け: JSONL 10 MiB / 1,000注釈、各 state 128 MiB、256層まで。state は要求時に読んで最大16記録をキャッシュします。起動中にファイルを更新したらサーバーを再起動してください

## 既存の採取 CLI

採取側には PyTorch と Transformers が必要です。既に用意したローカルモデルと、そのモデルに適した依存環境を使ってください。Inspector のデモにはどちらも不要です。

```sh
python symbolic_viewer.py \
  --model_path /path/to/local/model --offline \
  --symbols moon --capture_hidden \
  --save_states_dir states --annotations_path annotations.jsonl
```

各文脈のプロンプトを入力し、**生成部分に現れた** `moon` の token index に `literal` / `metaphorical` / `mythical` などのタグとメモを付けます。単語が複数 token に分かれる場合は注釈対象を確認してください。prompt 中だけの語は現行 CLI の token 注釈対象外です。segment 平均は `--span_annotation`、attention はモデルが対応する場合のみ `--capture_attn` を追加します。

**観察の限界:** 採取 CLI はテキストを生成した後、全文をもう一度 forward して hidden / attention を得ます。ライブ生成時の記録、思考の可視化、象徴的意味の証明ではありません。hidden index 0 は通常 embedding 出力です。モデル仕様によって解釈を確認してください。

## 開発・テスト

```sh
python -m unittest discover -s tests -v
python -m compileall -q trace_inspector.py tests
node --check inspector/inspector.js  # Node がある場合
node --test tests/test_inspector_ui.cjs  # DOM契約のテスト。描画テストではありません
```

実モデルはダウンロードしません。テスト内の小さな数値ベクトルは計算の検証用で、デモや実測データとして表示しません。PyTorch がある場合だけ実 `.pt` round-trip テストも走ります。

この Inspector は main の `1ae4db9` を基点に別ファイルとして追加しています。既存の採取スクリプトは変更していません。[PR #5](https://github.com/RyoSpiralArchitect/symbolic-association-obsever/pull/5) の missing-attention 修正は独立の作業で、この変更には取り込んでいません。
