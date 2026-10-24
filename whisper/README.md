# octantvector-whisper

OctantVector のビュー `whisper_language`（Whisper large-v3 で判定した、歌の言語）。使い方は [OctantVectorPlugins の README](../README.md) にあります。

- 曲全体を、曲の長さ ÷ 30 秒（切り上げ）か所の 30 秒で、始まりから終わりまで等間隔に覆い（箇所どうしは少しずつ均等に重なり、曲のどこかだけを二重に数えることはない。30 秒より短い曲は曲全体の 1 か所）、それぞれ 16 kHz にして `<|startoftranscript|>` の次の予測を 100 個の言語トークンだけで softmax します（文字起こしはしません）
- 言語の確率のエントロピーが高い箇所、歌のない曲が出やすい言語（km など 18 個）になった箇所、1 位の確率が 0.5 未満の箇所は、歌なしとします
- ベクトルは 100 次元で、歌のある箇所のうち各言語の箇所の数を L2 正規化したもの、`norm` は `none` です。同じ 1 つの言語の曲どうしの cos は 1、違う言語なら 0、2 言語の曲はその間になります。歌のある箇所がない曲はベクトルなしです
- 1 か所（30% の位置）だけでは前奏や間奏に当たり、曲全体を見た場合に言語が付く曲の 82% しか拾えませんでした（2,000 件、開発用の PC、2026-10-20）
- このパッケージのコード: MIT（LICENSE）
- モデル [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)（revision `06f233fe06e710322aca913c1bc4249a0d71fce1`）のライセンス: モデルカードでは Apache-2.0。モデルのファイルはこのパッケージに含まれず、`octantharvest models` が `models_dir` に置きます。
