# octantvector-wavlm

OctantVector のビュー `vocal_wavlm_sv`（WavLM base plus の話者照合 x-vector、歌声の分離後）。使い方は [OctantVectorPlugins の README](../README.md) にあります。歌声の分離に `octantvector[vocal]` が要るので `[vocal]` 付きで入れます。

- このパッケージのコード: MIT（LICENSE）
- モデル [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv)（revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`）のライセンス: CC BY-SA 3.0（モデルカードがリンクする [UniSpeech の LICENSE](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)。2026-10-23 確認）。モデルのファイルはこのパッケージに含まれず、`octantharvest models` が `models_dir` に置きます。
