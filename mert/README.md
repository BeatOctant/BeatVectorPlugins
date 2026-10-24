# octantvector-mert

OctantVector のビュー `mert_v1_95m`、`vocal_mert_v1_95m`（MERT-v1-95M）。使い方は [OctantVectorPlugins の README](../README.md) にあります。

- ライセンス: CC BY-NC 4.0（LICENSE、NOTICE。写したコードの上流のライセンス文は `licenses/`）。モデルのコード（`modeling_MERT.py`、`configuration_MERT.py`）を [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)（revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`）から写して含むため、このパッケージ全体をそのライセンスにしています。非営利でのみ使えます。
- モデルはこの写しのコードで組み立てます。モデルのフォルダにある `.py` は読まず、取ってもきません（`trust_remote_code` は使いません）。重みは `octantharvest models` が `models_dir` に置きます。
