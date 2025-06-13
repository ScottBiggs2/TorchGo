import math
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# BOARD_SIZE = 9 # 19 for full size
# NUM_MOVES = BOARD_SIZE * BOARD_SIZE  # 361

class ResidualBlock(
    nn.Module):  # Residual block, modelled after ResNets: https://en.wikipedia.org/wiki/Residual_neural_network
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # Why? Regularization effect of BN might gloss over subtle but extremely important feats.
        # self.norm1 = nn.LayerNorm(out_channels)
        # self.norm2 = nn.LayerNorm(out_channels)

        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        residual = x
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        if self.out_channels == self.in_channels:
            h += residual

        out = h
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, BOARD_SIZE):
        super().__init__()
        self.BOARD_SIZE = BOARD_SIZE

        # Shared convolutional backbone
        # to do - expand and add deep skips
        self.block_1 = ResidualBlock(8, 64) # changed from 2 when adding gaussian field support
        self.block_2 = ResidualBlock(64, 128)
        self.block_3 = ResidualBlock(128, 128)
        self.block_4 = ResidualBlock(128, 64)

        # Policy head: a 1×1 conv to 1 channel → flatten → softmax[361]
        self.policy_block = ResidualBlock(64, 64)
        self.conv_policy = nn.Conv2d(64, 1, kernel_size=1)
        self.fc_policy_1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.fc_policy_2 = nn.Linear(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value head: a 1×1 conv to 1 channel → flatten → FC to 64 → FC to 1 → tanh
        self.value_block = ResidualBlock(64, 64)
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
        x = F.elu(self.block_1(x)) # swapped relu for elu
        x = F.elu(self.block_2(x))
        x = F.elu(self.block_3(x))
        x = F.elu(self.block_4(x))

        p = F.elu(self.policy_block(x))
        p = self.conv_policy(p)  # [B,1,19,19]
        p = p.view(x.shape[0], -1)  # [B,361]
        p = F.elu(self.fc_policy_1(p))  # [B,361]
        policy = F.softmax(self.fc_policy_2(p), dim=1)  # [B,361]

        # ----- Value head -----
        v = F.elu(self.value_block(x))  # [B,64,19,19]
        v = self.conv_value(v)  # [B,1,19,19]
        v = v.view(x.shape[0], -1)  # [B,361]
        v = F.elu(self.fc_value1(v))  # [B,64]
        logit = self.fc_value2(v)  # [B,1]
        value = torch.tanh(logit)
        # +1 Black is winning -1 White is winning

        return policy, value
