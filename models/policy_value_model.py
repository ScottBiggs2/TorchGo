import math
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# BOARD_SIZE = 9 # 19 for full size
# NUM_MOVES = BOARD_SIZE * BOARD_SIZE  # 361

class PolicyValueNet(nn.Module):
    def __init__(self, BOARD_SIZE):
        super().__init__()
        self.BOARD_SIZE = BOARD_SIZE
        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Policy head: a 1×1 conv to 1 channel → flatten → softmax[361]
        self.conv_policy = nn.Conv2d(64, 1, kernel_size=1)

        # Value head: a 1×1 conv to 1 channel → flatten → FC to 64 → FC to 1 → tanh
        self.conv_value = nn.Conv2d(64, 1, kernel_size=1)
        self.fc_value1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
          - x: [B, 2, 19, 19], dtype=float32
        Returns:
          - policy: [B, 361] softmax probabilities over all board moves (flattened row‐major)
          - value:  [B, 1] tanh‐activated scalar in [-1, +1]
        """
        # Shared conv layers
        x = F.relu(self.conv1(x))    # → [B,32,19,19]
        x = F.relu(self.conv2(x))    # → [B,64,19,19]

        # Policy head
        p = self.conv_policy(x)      # → [B, 1, 19, 19]
        p = p.view(x.shape[0], -1)   # → [B, 361]
        policy = F.softmax(p, dim=-1)

        # Value head
        v = self.conv_value(x)       # → [B, 1, 19, 19]
        v = v.view(x.shape[0], -1)   # → [B, 361]
        v = F.relu(self.fc_value1(v))  # → [B, 64]
        v = torch.tanh(self.fc_value2(v))  # → [B, 1]

        return policy, v
