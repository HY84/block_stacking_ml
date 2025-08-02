# scripts/evaluate.py

import sys
import os
import torch
import numpy as np

# 親ディレクトリをPythonの検索パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import get_dataloaders
from src.model import PlacementModel

# --- 設定 ---
CSV_PATH = "data/01_raw/stacking_data.csv"
BATCH_SIZE = 64
MODEL_PATH = "outputs/trained_models/placement_model_20250802_084113.pth"
NUM_BLOCK_TYPES = 4

def main():
    _, val_loader = get_dataloaders(CSV_PATH, BATCH_SIZE)
    if val_loader is None:
        print("検証データが読み込めません。処理を終了します。")
        return
    print("検証用データを準備しました。")

    model = PlacementModel(
        input_size=15,
        num_slots=13,
        num_rotation_classes=3
    )
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"エラー: 学習済みモデルが見つかりません: {MODEL_PATH}")
        return
        
    model.eval()
    print("学習済みモデルを読み込みました。")

    correct_slots = 0
    correct_rotations = 0
    total = 0
    sample_index = 0
    
    # data_utils.pyのエンコーディング順に合わせる
    # block_names = ['T', 'O', 'I', 'L']
    block_names = ['T', 'L', 'I', 'O']

    print("\n--- 間違いの詳細 ---")
    
    with torch.no_grad():
        for features, (labels_slot, labels_rotation) in val_loader:
            outputs_slot, outputs_rotation = model(features)

            # ▼▼▼ MLP用のマスキング処理 ▼▼▼
            # 1. 入力データからboard_state部分を取得
            board_state = features[:, :13]
            # 2. board_stateが-1のスロットをマスクする
            mask = (board_state == -1).float() * -1e9
            # 3. モデルの出力にマスクを適用
            masked_outputs_slot = outputs_slot + mask
            
            _, predicted_slot = torch.max(masked_outputs_slot.data, 1)
            _, predicted_rotation = torch.max(outputs_rotation.data, 1)
            
            for i in range(len(labels_slot)):
                label_s = labels_slot[i].item()
                pred_s = predicted_slot[i].item()
                label_r = labels_rotation[i].item()
                pred_r = predicted_rotation[i].item()

                is_slot_correct = (label_s == pred_s)
                is_rotation_correct = (label_r == pred_r)

                if not is_slot_correct or not is_rotation_correct:
                    # ▼▼▼ 観測（入力）情報を取得して表示する処理を追加 ▼▼▼
                    
                    # 1. 特徴量ベクトルから盤面状態とブロック種類を分離
                    board_state_tensor = features[i][:11]
                    block_type_tensor = features[i][11:]
                    
                    # 2. データを人間が読みやすい形式に戻す
                    #    (正規化を元に戻し、整数に)
                    # board_state_numpy = board_state_tensor.numpy()*20
                    board_state_numpy = board_state_tensor.numpy()
                    board_state_list = [int(round(x)) for x in board_state_numpy]
                    
                    block_type_index = torch.argmax(block_type_tensor).item()
                    block_type_name = block_names[block_type_index]

                    print(f"サンプル #{sample_index}:")
                    print(f"  観測: 盤面={board_state_list}, ブロック='{block_type_name}'")
                    print(f"  正解: [スロット={label_s}, 回転={label_r}]")
                    print(f"  予測: [スロット={pred_s}, 回転={pred_r}] <-- ❌ 間違い")
                
                sample_index += 1
            
            total += labels_slot.size(0)
            correct_slots += (predicted_slot == labels_slot).sum().item()
            correct_rotations += (predicted_rotation == labels_rotation).sum().item()

    if total == 0:
        print("評価データがありません。")
        return

    slot_accuracy = 100 * correct_slots / total
    rotation_accuracy = 100 * correct_rotations / total
    
    print("\n--- 評価結果 ---")
    print(f"スロット配置の正解率: {slot_accuracy:.2f} %")
    print(f"回転の正解率: {rotation_accuracy:.2f} %")
    print("----------------")

if __name__ == '__main__':
    main()