# octantvector-siglip

OctantVector のビュー `siglip_cover`（SigLIP so400m で埋め込んだ、マップのカバー画像）。使い方は [OctantVectorPlugins の README](../README.md) にあります。

- カバーは Info.dat が名指すファイル（なければ zip の中の最初の画像）です。768 px に収まるよう縮めてから、SigLIP の画像の前処理（384×384、縦横比は保たない）にかけます
- ベクトルは画像の埋め込み（アテンションプーリングの頭の出力）を L2 正規化したもので、`norm` は `none` です。同じ絵のカバーどうしの cos は 1 近くになります
- このパッケージのコード: MIT（LICENSE）
- モデル [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)（revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`）のライセンス: モデルカードでは Apache-2.0。モデルのファイルはこのパッケージに含まれず、`octantharvest models` が `models_dir` に置きます。
