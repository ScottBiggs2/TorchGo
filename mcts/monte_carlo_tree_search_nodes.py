import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
# from training.self_play_system import generate_influence_fields

# make sure this is the same version as the one in self_play_system
def generate_influence_fields(stone_tensor: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    """
    Input:  stone_tensor of shape (bs, 4, 19, 19)
           - channels 0,1: current board (black, white)
           - channels 2,3: previous board (black, white)
    Output: influence_tensor of shape (bs, 4, 19, 19)
           - channels 0,1: influence fields for current board
           - channels 2,3: influence fields for previous board
    """
    bs, ch, h, w = stone_tensor.shape
    assert ch == 4, "Expected 4 input channels (current black/white, previous black/white)"

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

    for i in range(ch):  # current black, current white, previous black, previous white
        influence[:, i:i+1] = F.conv2d(
            stone_tensor[:, i:i+1],  # shape (bs,1,19,19)
            kernel, padding=kernel_size // 2
        )

    return influence


class MCTSNode:
    __slots__ = (
        "game", "parent", "move",
        "children",  # move → MCTSNode
        "P",  # move → prior prob
        "N",  # move → visit count
        "W",  # move → total value
        "Q",  # move → mean value
        "visits"  # int: number of times this node itself was visited
    )

    def __init__(self, game: GoGame, parent: Optional["MCTSNode"] = None,
                 move: Optional[Tuple[int, int]] = None):
        self.game = game
        self.parent = parent
        self.move = move

        self.children = {}  # { move: MCTSNode }
        self.P = None  # to be filled by network on first expansion
        self.N = {}  # { move: int }
        self.W = {}  # { move: float }
        self.Q = {}  # { move: float }
        self.visits = 0  # how many times this node was visited

    def is_fully_expanded(self) -> bool:
        """
        True if every legal move (including pass) from this position already has a child.
        We check by comparing the number of moves in self.P to len(self.children).
        """
        if self.P is None:
            return False
        return len(self.children) == len(self.P)

    def expand_and_evaluate(self, net: PolicyValueNet, device: torch.device) -> float:
        """
        1) Build a tensor input [1,2,19,19] from self.game (current + previous board),
        2) Call net to get (policy_probs [1,361+1], value [1,1]),
        3) Mask out illegal moves, renormalize priors,
        4) Store self.P = { (x,y) → prob, None→prob_pass },
           initialize self.N[...] = 0, self.W[...] = 0, self.Q[...] = 0 for all keys in self.P,
        5) Return the scalar 'value' as a Python float.
        """
        # 1) Prepare input tensor
        current = self.game.board.clone().unsqueeze(0).float().to(device)  # [1,19,19]
        if self.game.history:
            prev = self.game.history[-1].float().unsqueeze(0).to(device)  # [1,19,19]
        else:
            prev = torch.zeros_like(current)  # [1,19,19]

        # Convert to binary planes for each color
        BLACK = self.game.BLACK
        WHITE = self.game.WHITE
        
        # Current board
        current_black = (current == BLACK).to(torch.float32)  # 1.0 where Black stones
        current_white = (current == WHITE).to(torch.float32)  # 1.0 where White stones
        
        # Previous board
        prev_black = (prev == BLACK).to(torch.float32)  # 1.0 where Black stones
        prev_white = (prev == WHITE).to(torch.float32)  # 1.0 where White stones
        
        # Stack current and previous states
        state_tensor = torch.stack([current_black, current_white, prev_black, prev_white], dim=1)  # [1,4,19,19]
        state_tensor = torch.concat([state_tensor,
                                     generate_influence_fields(state_tensor, sigma=1),
                                     generate_influence_fields(state_tensor, sigma=3),
                                     generate_influence_fields(state_tensor, sigma=6)], dim=1)
        #[1, 16, 19, 19]

        # 2) Forward pass
        with torch.no_grad():
            policy_logits, value = net(state_tensor)  # policy_logits: [1,361+1], value: [1,1]

        policy_logits = policy_logits.squeeze(0).cpu()  # → [361+1]
        value = value.item()  # scalar

        # 3) Build a mask of legal moves
        legal_mask = torch.zeros(self.game.BOARD_SIZE**2 + 1, dtype=torch.bool)  # +1 for pass
        for idx in range(self.game.BOARD_SIZE**2):
            x, y = divmod(idx, self.game.BOARD_SIZE)
            if self.game.is_legal(x, y):
                legal_mask[idx] = True
        # "Pass" is always legal
        legal_mask[-1] = True

        # 4) Extract raw priors for legal moves
        priors = {}
        if legal_mask.any():
            # Grab the logits for legal indices, then renormalize
            legal_logits = policy_logits[legal_mask]  # shape = [num_legal]
            legal_probs = legal_logits / legal_logits.sum().clamp(min=1e-8)  # normalize
            legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(1).tolist()
            for i, idx in enumerate(legal_indices):
                if idx == self.game.BOARD_SIZE**2:  # Pass move
                    priors[None] = legal_probs[i].item()
                else:
                    x, y = divmod(idx, self.game.BOARD_SIZE)
                    priors[(x, y)] = legal_probs[i].item()
        else:
            # No legal board moves? Must pass
            priors[None] = 1.0

        # 5) Store P, and initialize N,W,Q
        self.P = priors
        for move in priors:
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0

        return value

    def select_child(self, c_puct: float) -> Tuple[Optional[Tuple[int, int]], "MCTSNode"]:
        """
        From self.P, pick move a that maximizes:
          U(s,a) = Q[s,a] + c_puct * P[s,a] * sqrt(self.visits) / (1 + N[s,a])
        If that move has already been expanded, return its child.
        Otherwise, create a new child node by cloning self.game and playing a.
        """
        best_score = -float("inf")
        best_move: Optional[Tuple[int, int]] = None

        total_visits = self.visits
        sqrt_visits = math.sqrt(total_visits) if total_visits > 0 else 0.0

        for mv, prior in self.P.items():
            n_sa = self.N[mv]
            q_sa = self.Q[mv]
            u_sa = q_sa + c_puct * prior * (sqrt_visits / (1 + n_sa))
            if u_sa > best_score:
                best_score = u_sa
                best_move = mv

        # best_move is now the action to descend
        if best_move in self.children:
            return best_move, self.children[best_move]

        # Otherwise, expand it on the fly
        new_game = self.game.clone()
        if best_move is None:
            new_game.play_move()  # pass
        else:
            new_game.play_move(best_move[0], best_move[1])

        child = MCTSNode(new_game, parent=self, move=best_move)
        self.children[best_move] = child
        return best_move, child

