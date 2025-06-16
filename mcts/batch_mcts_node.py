import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet

def generate_influence_fields(stone_tensor: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    """
    Input:  stone_tensor of shape (bs, 6, h, w)
           - channels 0,1: current board (black, white)
           - channels 2,3: previous board (black, white)
           - channels 4,5: previous-previous board (black, white)
    Output: influence_tensor of shape (bs, 6, h, w)
           - channels 0,1: influence fields for current board
           - channels 2,3: influence fields for previous board
           - channels 4,5: influence fields for previous-previous board
    """
    bs, ch, h, w = stone_tensor.shape
    assert ch == 6, "Expected 6 input channels (current black/white, previous black/white, prev-prev black/white)"

    # Build 2D Gaussian kernel
    kernel_size = int(6 * sigma) | 1  # make it odd
    coords = torch.arange(kernel_size) - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing="ij")
    gaussian_kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize
    kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,K,K)

    # Prepare to convolve each color channel independently
    kernel = kernel.to(stone_tensor.device)
    influence = torch.zeros_like(stone_tensor)

    for i in range(ch):  # current black/white, previous black/white, prev-prev black/white
        influence[:, i:i+1] = F.conv2d(
            stone_tensor[:, i:i+1],  # shape (bs,1,h,w)
            kernel, padding=kernel_size // 2
        )

    return influence

class MCTSNode:
    __slots__ = (
        "game", "parent", "move", "children", "P", "N", "W", "Q", 
        "visits", "state_tensor", "virtual_loss"
    )

    def __init__(self, game: GoGame, parent: Optional["MCTSNode"] = None,
                 move: Optional[Tuple[int, int]] = None):
        self.game = game
        self.parent = parent
        self.move = move

        self.children: Dict[Optional[Tuple[int, int]], "MCTSNode"] = {}
        self.P: Optional[Dict[Optional[Tuple[int, int]], float]] = None
        self.N: Dict[Optional[Tuple[int, int]], int] = defaultdict(int)
        self.W: Dict[Optional[Tuple[int, int]], float] = defaultdict(float)
        self.Q: Dict[Optional[Tuple[int, int]], float] = defaultdict(float)
        self.visits = 0
        self.state_tensor = None
        self.virtual_loss: Dict[Optional[Tuple[int, int]], int] = defaultdict(int)

    def get_state_tensor(self, device: torch.device) -> torch.Tensor:
        """Cache and return state tensor for network input"""
        if self.state_tensor is not None:
            return self.state_tensor
            
        BOARD_SIZE = self.game.BOARD_SIZE
        BLACK = self.game.BLACK
        WHITE = self.game.WHITE
        
        # Get current board state
        current = self.game.board.clone().float().to(device)
        
        # Get previous board states (t-1 and t-2)
        if len(self.game.history) >= 2:
            prev = self.game.history[-1].float().to(device)  # t-1
            prev_prev = self.game.history[-2].float().to(device)  # t-2
        elif len(self.game.history) == 1:
            prev = self.game.history[-1].float().to(device)  # t-1
            prev_prev = torch.zeros_like(current)  # t-2 (empty)
        else:
            prev = torch.zeros_like(current)  # t-1 (empty)
            prev_prev = torch.zeros_like(current)  # t-2 (empty)
        
        # Create individual channels
        current_black = (current == BLACK).float()
        current_white = (current == WHITE).float()
        prev_black = (prev == BLACK).float()
        prev_white = (prev == WHITE).float()
        prev_prev_black = (prev_prev == BLACK).float()
        prev_prev_white = (prev_prev == WHITE).float()
        
        # Stack channels and add batch dimension
        base_tensor = torch.stack([
            current_black, current_white, 
            prev_black, prev_white,
            prev_prev_black, prev_prev_white
        ], dim=0).unsqueeze(0)
        
        # Generate influence fields at multiple scales
        influence_fields = []
        for sigma in [1, 3, 6]:
            influence = generate_influence_fields(base_tensor, sigma)
            influence_fields.append(influence)
        
        # Combine base tensor with influence fields
        self.state_tensor = torch.cat([base_tensor] + influence_fields, dim=1)
        return self.state_tensor

    def select_child(self, c_puct: float) -> Tuple[Optional[Tuple[int, int]], "MCTSNode"]:
        best_score = -float("inf")
        best_move = None
        parent_visits = self.visits
        sqrt_parent_visits = math.sqrt(parent_visits) if parent_visits > 0 else 1.0

        # print(f"DEBUG: Selecting child with {len(self.P)} moves available")
        for move, prior in self.P.items():
            n_sa = self.N[move]
            q_sa = self.Q[move]
            
            # Apply virtual loss to exploration term
            virtual_loss = self.virtual_loss.get(move, 0)
            
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # Add small epsilon to denominator to avoid division by zero
            u_sa = q_sa + c_puct * prior * sqrt_parent_visits / (1 + n_sa + virtual_loss + 1e-8)
            
            # print(f"DEBUG: Move {move} - N: {n_sa}, Q: {q_sa}, virtual_loss: {virtual_loss}, UCB: {u_sa}")
            
            if u_sa > best_score:
                best_score = u_sa
                best_move = move

        # If we already have this child, return it
        if best_move in self.children:
            # Apply virtual loss when selecting
            self.virtual_loss[best_move] += 1
            # print(f"DEBUG: Reusing existing child for move {best_move}, virtual_loss now {self.virtual_loss[best_move]}")
            return best_move, self.children[best_move]

        # Only create a new child if we don't have one for this move
        # print(f"DEBUG: Creating new child for move {best_move}")
        new_game = self.game.clone()
        if best_move is None:
            new_game.play_move()  # pass
        else:
            new_game.play_move(best_move[0], best_move[1])
        
        child = MCTSNode(new_game, self, best_move)
        self.children[best_move] = child
        self.virtual_loss[best_move] += 1  # Apply virtual loss
        # print(f"DEBUG: Created new child for move {best_move}, virtual_loss: {self.virtual_loss[best_move]}")
        return best_move, child

    def revert_virtual_loss(self, move):
        """Remove virtual loss after simulation completes"""
        if move in self.virtual_loss:
            self.virtual_loss[move] -= 1
            # print(f"DEBUG: Reverting virtual loss for move {move}, now {self.virtual_loss[move]}")
            if self.virtual_loss[move] == 0:
                del self.virtual_loss[move]
                # print(f"DEBUG: Removed virtual loss for move {move}")
