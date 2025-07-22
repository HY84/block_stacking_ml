# Dockerfile

FROM python:3.10-slim

# ↓↓↓ 作業ディレクトリをVSCodeが使うパスに合わせます ↓↓↓
WORKDIR /workspaces/block_stacking_ml

# 依存関係ファイルのみを先にコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# GUI表示のためのライブラリをインストール
RUN apt-get update && apt-get install -y libx11-6 libxext6 libxrender1 x11-apps
