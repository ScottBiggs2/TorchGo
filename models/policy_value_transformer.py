import math
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for Go board spatial information"""
    def __init__(self, d_model, max_len=19):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encodings for rows and columns
        pe_row = torch.zeros(max_len, d_model)
        pe_col = torch.zeros(max_len, d_model)
        position_row = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        position_col = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        exp_term = torch.arange(0, d_model, 2).float() * -1 * (math.log(10000.0) / d_model)
        div_term = torch.exp(exp_term)
        
        pe_row[:, 0::2] = torch.sin(position_row * div_term)
        pe_row[:, 1::2] = torch.cos(position_row * div_term)
        pe_col[:, 0::2] = torch.sin(position_col * div_term)
        pe_col[:, 1::2] = torch.cos(position_col * div_term)
        
        self.register_buffer('pe_row', pe_row)
        self.register_buffer('pe_col', pe_col)

    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        batch_size, seq_len, _ = x.size()
        board_size = int(math.sqrt(seq_len))
        
        # Create full 2D grid
        row_idx = torch.arange(board_size).repeat_interleave(board_size)
        col_idx = torch.arange(board_size).repeat(board_size)
        
        # Add row and column positional encodings
        pe = self.pe_row[row_idx] + self.pe_col[col_idx]
        return x + pe.unsqueeze(0).to(x.device)


class TransformerBlock(nn.Module):
    """GPT-style transformer block with causal masking"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal_mask=True):
        # Self-attention with causal masking
        attn_mask = None
        if causal_mask:
            seq_len = x.size(1)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        attn_output, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=attn_mask,
            need_weights=False
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class PolicyValueTransformer(nn.Module):
    """Transformer-based Go policy and value network"""
    def __init__(self, BOARD_SIZE, d_model=128, nhead=8, num_layers=12, dropout=0.1):
        super().__init__()
        self.BOARD_SIZE = BOARD_SIZE
        self.num_positions = BOARD_SIZE * BOARD_SIZE
        self.d_model = d_model
        
        # Input embedding: project 16 channels to d_model
        self.input_embedding = nn.Linear(24, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(d_model, max_len=BOARD_SIZE)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1))
        
        # Pass move head (special token)
        self.pass_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pass_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1))
        
        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Tanh())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
          x: [B, 16, H, W] - 16-channel board representation
        Returns:
          policy: [B, H*W+1] - move probabilities
          value: [B, 1] - position evaluation
        """
        batch_size = x.size(0)
        board_size = x.size(2)
        
        # Reshape to sequence: [B, H*W, 16]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, board_size*board_size, -1)
        
        # Embed inputs: [B, seq_len, d_model]
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Add pass token: [B, seq_len+1, d_model]
        pass_tokens = self.pass_token.expand(batch_size, -1, -1)
        x = torch.cat([x, pass_tokens], dim=1)
        
        # Transformer processing
        for block in self.transformer_blocks:
            x = block(x, causal_mask=True)
        
        # Policy head
        board_logits = self.policy_head(x[:, :-1])  # All board positions
        pass_logit = self.pass_head(x[:, -1:])      # Pass token
        policy_logits = torch.cat([board_logits, pass_logit], dim=1).squeeze(-1)
        policy = F.softmax(policy_logits, dim=1)
        
        # Value head uses global average pooling
        global_rep = x.mean(dim=1)  # [B, d_model]
        value = self.value_head(global_rep)
        
        return policy, value