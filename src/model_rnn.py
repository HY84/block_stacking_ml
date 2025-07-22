# src/model.py

import torch
import torch.nn as nn

class PlacementModel(nn.Module):
    def __init__(self, input_size=15, num_slots=12, num_rotation_classes=3):
        super(PlacementModel, self).__init__()
        
        # LSTM層：シーケンスデータを処理し、文脈を記憶する
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True # (バッチサイズ, シーケンス長, 特徴量数)の形式でデータを受け取る
        )
        
        # LSTMの出力を受ける中間層
        self.layer1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        
        # 出力層（ヘッド）は同じ
        self.slot_head = nn.Linear(64, num_slots)
        self.rotation_head = nn.Linear(64, num_rotation_classes)

    def forward(self, x):
        # xの形状: (バッチサイズ, シーケンス長, 特徴量数)
        
        # LSTM層を通過
        lstm_out, _ = self.lstm(x)
        
        # 最後の時点の出力だけを取り出す
        # lstm_outの形状: (バッチサイズ, シーケンス長, 隠れ層サイズ)
        # last_outputの形状: (バッチサイズ, 隠れ層サイズ)
        last_output = lstm_out[:, -1, :]
        
        # 中間層と出力層を通過
        x = self.relu(self.layer1(last_output))
        slot_output = self.slot_head(x)
        rotation_output = self.rotation_head(x)
        
        # ▼▼▼ This must return exactly two values ▼▼▼
        return slot_output, rotation_output