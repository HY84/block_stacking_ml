# src/model.py

import torch
import torch.nn as nn

class PlacementModel(nn.Module):
    def __init__(self, input_size=15, num_slots=13, num_rotation_classes=3):
        super(PlacementModel, self).__init__()
        
        # Shared hidden layers
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        
        # Output layer for slot prediction
        self.slot_head = nn.Linear(64, num_slots)
        # Output layer for rotation prediction
        self.rotation_head = nn.Linear(64, num_rotation_classes)

    def forward(self, x):
        # Pass through shared layers
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        # Calculate outputs from each head
        slot_output = self.slot_head(x)
        rotation_output = self.rotation_head(x)
        
        # ▼▼▼ This must return exactly two values ▼▼▼
        return slot_output, rotation_output