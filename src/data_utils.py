# src/data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

NUM_SLOTS = 11

class StackingDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # --- 特徴量 (X) の作成 (変更なし) ---
        board_state_cols = [f'z_{i}' for i in range(NUM_SLOTS)]
        board_states = df[board_state_cols].values.astype(np.float32) / 20.0
        block_types = pd.get_dummies(df['block_type']).reindex(columns=['T', 'L', 'I', 'O'], fill_value=0).values.astype(np.float32)
        self.X = torch.tensor(np.hstack([board_states, block_types]), dtype=torch.float32)

        # --- ラベル (y) を2種類作成 ---
        # 1. スロットのラベル
        # CSVの列名が 'placed_x' の場合は 'slot_index' にリネームしてから使用
        if 'placed_x' in df.columns:
            df.rename(columns={'placed_x': 'slot_index'}, inplace=True)
        self.y_slot = torch.tensor(df['slot_index'].values, dtype=torch.long)

        # 2. 回転のラベル (回転なし:0, 回転あり:1 の2値)
        rotations = df['rotation_id'].values
        # rotationが0なら0, それ以外(1,2,3)なら1に変換
        self.y_rotation = torch.tensor((rotations > 0).astype(int), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 2種類のラベルをタプルとして返す
        return self.X[idx], (self.y_slot[idx], self.y_rotation[idx])

def get_dataloaders(csv_path, batch_size=32):
    """
    データセット全体を読み込み、訓練用と検証用に分割してDataLoaderを返す
    """
    dataset = StackingDataset(csv_path)
    
    # データを訓練用(80%)と検証用(20%)にランダムに分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader