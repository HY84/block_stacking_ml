# scripts/train.py
"""python scripts/train.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import get_dataloaders
from src.model import PlacementModel
from datetime import datetime

# --- ハイパーパラメータ設定 ---
CSV_PATH = "data/01_raw/stacking_data.csv"
BATCH_SIZE = 64
EPOCHS = 200 # データセット全体を何回学習するか
LEARNING_RATE = 0.005
MODEL_SAVE_PATH = f"outputs/trained_models/placement_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

# scripts/train.py

def main():
    # 1. データの準備
    train_loader, val_loader = get_dataloaders(CSV_PATH, BATCH_SIZE)
    print("データ準備完了！")

    # 2. モデルの初期化
    model = PlacementModel()
    print("モデル初期化完了！")

    # 3. 損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 学習ループ (修正版) ---
    print("学習を開始します...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # ▼▼▼ ラベルを2種類受け取るように修正 ▼▼▼
        for features, (labels_slot, labels_rotation) in train_loader:
            optimizer.zero_grad()
            
            # ▼▼▼ モデルから2つの出力を受け取る ▼▼▼
            outputs_slot, outputs_rotation = model(features)
            
            # ▼▼▼ 損失をそれぞれ計算し、合計する ▼▼▼
            loss_slot = criterion(outputs_slot, labels_slot)
            loss_rotation = criterion(outputs_rotation, labels_rotation)
            loss = loss_slot + loss_rotation # 損失を合算
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"エポック [{epoch+1}/{EPOCHS}], 損失: {avg_loss:.4f}")

    print("学習が完了しました！")

    # 5. 学習済みモデルの保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"モデルを {MODEL_SAVE_PATH} に保存しました。")

if __name__ == '__main__':
    # outputs/trained_models フォルダがなければ作成
    import os
    os.makedirs("outputs/trained_models", exist_ok=True)
    main()