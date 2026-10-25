# OctantVectorPlugins

> **To use BeatOctant** (install, known issues): [Getting Started](https://github.com/BeatOctant/OctantSpecs/blob/main/GettingStarted_en.md) · [はじめかた（日本語）](https://github.com/BeatOctant/OctantSpecs/blob/main/GettingStarted_ja.md)
>
> BeatOctant is written mostly with AI coding agents (Anthropic's Claude). The maintainer decides what to build and reviews every change. It does not generate maps; it recommends existing maps from BeatSaver. ([作り方（AI の使い方）](https://github.com/BeatOctant/OctantSpecs/blob/main/README.md#作り方ai-の使い方))
>
> OctantVectorPlugins adds audio and image model views to OctantVector, the engine of BeatOctant, a Beat Saber map recommender (the OctantPath mod for Beat Saber PC and OctantPath for Quest).
>
> BeatOctant is an unofficial fan project for Beat Saber. It is not affiliated with, endorsed by, or sponsored by Beat Games or Meta. Beat Saber and Meta Quest are trademarks of their respective owners. Beat Games・Meta とは関係のない非公式の MOD・ツールです。
>
> 報告と案は Discord の BeatOctant サーバー（バグは `#bug-reports`、案は `#feedback-ideas`）で受けます。外からの pull request は受けていません。

[OctantVector](https://github.com/BeatOctant/OctantVector) に音や画像のモデルを使うビューを足すプラグインです。系統ごとに一つのフォルダで、それぞれが独立したパッケージ（自分の LICENSE 付き）です。BeatOctant 全体の中での位置は [OctantSpecs の概要](https://github.com/BeatOctant/OctantSpecs#概要)にあります。

| フォルダ | パッケージ | ビュー | ライセンス |
|---|---|---|---|
| `clap/` | `octantvector-clap` | `clap_htsat_unfused`、`vocal_clap_htsat_unfused`、`clap_concepts` | MIT（モデルは Apache-2.0） |
| `mert/` | `octantvector-mert` | `mert_v1_95m`、`vocal_mert_v1_95m` | CC BY-NC 4.0（モデルのコードの写しを含むため。`mert/NOTICE`） |
| `wavlm/` | `octantvector-wavlm` | `vocal_wavlm_sv` | MIT（モデルは CC BY-SA 3.0） |
| `whisper/` | `octantvector-whisper` | `whisper_language`（歌の言語。歌のある箇所の言語の割合） | MIT（モデルはモデルカードでは Apache-2.0） |
| `siglip/` | `octantvector-siglip` | `siglip_cover`（カバー画像） | MIT（モデルはモデルカードでは Apache-2.0） |

リポジトリ全体のライセンスはありません。直下の `LICENSE` は各フォルダの LICENSE への案内です。

`vocal_` のビューは Demucs で分けた歌声を使うので、`[vocal]` 付きで入れます（`octantvector[vocal]` が入ります）。

## 組み合わせる版

支えるのは、各部品の最新の版どうしの組み合わせだけです（[decisions.md の全体と段階](https://github.com/BeatOctant/OctantSpecs/blob/main/decisions.md#全体と段階)、2026-10-22）。この README を書いた時点の相手は次のとおりです。

| 相手 | 版 | つなぐもの |
| --- | --- | --- |
| OctantVector | v0.9.2（各パッケージの範囲は `>=0.9.0,<0.10`） | `octantvector.plugin`（`PLUGIN_API` 0.3） |
| OctantHarvest（プラグインで DB を作る） | 最新 | OctantVector を通して呼ばれる |

## プラグイン API は不安定です

プラグインは `octantvector.plugin` だけを使います（`PLUGIN_API` 0.3。OctantVector 0.8 系のもの）。この API はまだ固まっておらず、変わるたびに `PLUGIN_API` の番号が上がります。各パッケージは書かれた番号を `OCTANTVECTOR_PLUGIN_API` に持ち、OctantVector の番号と違うと読み込まれません（そのビューを頼んだときに理由が出ます）。OctantVector を上げたら、同じ番号のプラグインに上げてください。

各ビューは、モデルの出所と条件を `credits` に持ちます。OctantVector が DB を書くとき、マニフェストのそのビューの `credits` に入ります（ボーカルのビューには Demucs の分も足されます）。

## 入れ方

OctantVector と同じ環境に、使うものだけ入れます。

```bash
pip install "octantvector-clap[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=clap"
pip install "octantvector-mert[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=mert"
pip install "octantvector-wavlm[vocal] @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=wavlm"
pip install "octantvector-whisper @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=whisper"
pip install "octantvector-siglip @ git+https://github.com/BeatOctant/OctantVectorPlugins#subdirectory=siglip"
```

このリポジトリにはまだ版のタグがないので、main から入れます。入れるとエントリポイント `octantvector.encoders` で OctantVector に登録されます。

## モデルのファイル

プラグインも OctantVector も、モデルを取りに行きません。各ビューは使うモデル（Hugging Face のリポジトリと固定した revision、ファイル名）を宣言し、`models_dir` の下の `hf/<repo>/<revision>/` からだけ読みます。ファイルは OctantHarvest の `octantharvest models` が置きます（契約1 の `models(views)` で何が要るかを聞きます）。

## 開発とテスト

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e "/path/to/OctantVector[embed]" -e "./clap[dev]" -e "./mert[dev]" -e "./wavlm[dev]" -e "./whisper[dev]" -e "./siglip[dev]"
.venv/bin/python -m pytest -q clap/tests mert/tests wavlm/tests whisper/tests siglip/tests
```

テストはネットワークにも本物の重みにも触れません。小さな設定でランダムに初期化したモデルを一時的な `models_dir` に置き、OctantVector 0.3 の時代（`PLUGIN_API` の 0.3 とは別の、OctantVector 本体の古い版）の計算（テストに写してあります）と同じベクトルになることを確かめます。歌声のビューは分離器を決まった偽物に替えて、クリップの切り方・歌声のある曲の判定・平均の取り方を比べます。`whisper_language` と `siglip_cover` は 0.3 の時代の計算がないので、小さなモデルで形と決まり（切り出す位置、言語なしにする条件、言語の割合、正規化）を確かめます。本物の重みでの値は、開発用の PC で Spike（2026-10-19）の値と比べます。

### CI

`.github/workflows/test.yml` がプルリクエストでテストを動かします。

## ライセンス

パッケージ（フォルダ）ごとに違います。リポジトリの直下には LICENSE がありません。

| フォルダ | ライセンス |
|---|---|
| `clap/` | [MIT](clap/LICENSE) |
| `mert/` | [CC BY-NC 4.0](mert/LICENSE)（非営利。モデルのコードの写しを含むため。[NOTICE](mert/NOTICE)） |
| `wavlm/` | [MIT](wavlm/LICENSE) |
| `whisper/` | [MIT](whisper/LICENSE) |
| `siglip/` | [MIT](siglip/LICENSE) |

モデルの重みの条件は、上の表とモデルカードを見てください。
