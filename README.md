# Block Stacking ML

## 概要

Block Stacking MLは、ブロック積み上げゲームの盤面状態・操作データを収集し、機械学習モデルで最適なブロック配置・回転を予測するプロジェクトです。  
GUIでデータ収集を行い、PyTorchを用いてモデルの学習・評価を行います。

### 現状の学習結果
BATCH_SIZE = 64\
EPOCHS = 200 \
LEARNING_RATE = 0.005\

--- 評価結果 ---
スロット配置の正解率: 86.67 %\
回転の正解率: 93.33 %


### 誤答の例
  観測: 盤面=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], ブロック='L'\
  正解: [スロット=3, 回転=0]\
  予測: [スロット=7, 回転=0] <-- ❌ 間違い

初期配置で、複数の配置がある場合。それほど重要な問題ではない。

### 懸念点
[現実で倒れそうな配置はリセットしてデータを集めた](data/PXL_20250719_101015004.jpg)\
[少数のブロックでも配置不可の場合がある](data/PXL_20250719_101202944.jpg)
---

## ディレクトリ構成

```
.
├── configs/                # ブロック定義(JSON)
├── data/                   # 収集データ・画像
│   ├── 01_raw/             # 生データ(CSV)
│   └── 02_processed/       # 前処理済みデータ
├── outputs/                # 学習済みモデル・ログ
│   ├── trained_models/     # 通常モデル
│   └── trained_models_RNN/ # RNNモデル
├── scripts/                # 実行スクリプト
├── src/                    # Pythonモジュール群
└── Dockerfile, docker-compose.yml, .devcontainer/
```

---

## 目的

- ブロック積み上げ盤面の状態抽象化・特徴量化
- 人間の操作データ収集（GUI）
- 機械学習による配置・回転予測モデルの構築
- RNNによる時系列予測もサポート

---

## セットアップ

### 1. Docker環境

本プロジェクトはVSCode Dev Container/Dockerで動作します。

```sh
# VSCodeで「Dev Container: Open Folder in Container」を実行
# もしくは手動でビルド
docker-compose build
docker-compose up
```

### 2. 依存パッケージ

Pythonパッケージは`requirements.txt`で管理されています。

- pygame
- torch
- numpy
- pandas

---

## 使い方

### 1. データ収集GUI

ブロック配置操作を記録し、CSVデータを生成します。

```sh
python scripts/generate_data.py
```

- 操作方法は画面右下に表示
- リセット操作も記録されます

### 2. モデル学習

収集したデータでモデルを学習します。

```sh
python scripts/train.py
```

- 学習済みモデルは`outputs/trained_models/`に保存

### 3. モデル評価

学習済みモデルの精度を検証します。

```sh
python scripts/evaluate.py
```

- 配置・回転の正解率が表示されます
- 間違いの詳細も出力されます

---

## 主要スクリプト・モジュール

- [`scripts/generate_data.py`](scripts/generate_data.py): データ収集GUI（通常盤面）
- [`scripts/train.py`](scripts/train.py): モデル学習（通常モデル）
- [`scripts/evaluate.py`](scripts/evaluate.py): モデル評価
- [`src/environment.py`](src/environment.py): 盤面・GUIロジック（通常）
- [`src/model.py`](src/model.py): 配置予測モデル
- [`src/data_utils.py`](src/data_utils.py): データセット・DataLoader

---

## データ仕様

- 盤面状態は監視ポイントごとの高さで特徴量化
- ブロック種類はワンホットエンコーディング
- 配置位置・回転IDも記録
- CSV形式で保存

---

## 開発環境

- OS: Debian GNU/Linux 12 (bookworm)（Dockerコンテナ）
- VSCode Dev Container対応
- X11 GUI（pygame）

---

## ライセンス

MIT License

---

## 補足

- configs/blocks.json, blocks_copy.jsonでブロック形状・色・回転定義
- DockerでX11 GUIを表示するため、`DISPLAY=:1`や`/tmp/.X11-unix`のマウントが必要
- 詳細は各スクリプト・モジュールのコメント参照
