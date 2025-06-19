from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.run_batched_mcts import run_batched_mcts
from mcts.batch_mcts_node import MCTSNode
from mcts.batch_mcts_node import generate_influence_fields



Example = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, examples: List[Example]):
        """
        Add a list of (state, pi, z) triples to the buffer.
        If over capacity, the oldest examples are discarded automatically.
        """
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int) -> List[Example]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

# Return to this later to give PolicyValueNet the game at time t (present) and time t-1 for Ko rules
def state_to_tensor(game: GoGame, device: torch.device) -> torch.Tensor:
    """
    Convert the current and previous positions into a [18,19,19] float32 tensor:
      Base channels (6):
      - channels 0,1: Current board - Black stones = 1.0, White stones = 0.0, Empty = 0.0
      - channels 2,3: Previous board (t-1) - Black stones = 1.0, White stones = 0.0, Empty = 0.0
      - channels 4,5: Previous-previous board (t-2) - Black stones = 1.0, White stones = 0.0, Empty = 0.0
    """
    current = game.board.float().to(device)  # shape [19,19], values ∈ {-1,0,+1}
    
    # Get previous board states if they exist, otherwise use zeros
    if len(game.history) >= 2:
        prev = game.history[-1].float().to(device)  # t-1
        prev_prev = game.history[-2].float().to(device)  # t-2
    elif len(game.history) == 1:
        prev = game.history[-1].float().to(device)  # t-1
        prev_prev = torch.zeros_like(current)  # t-2 (empty)
    else:
        prev = torch.zeros_like(current)  # t-1 (empty)
        prev_prev = torch.zeros_like(current)  # t-2 (empty)

    # Convert to binary planes for each color
    BLACK = game.BLACK
    WHITE = game.WHITE
    
    # Current board
    current_black = (current == BLACK).to(torch.float32)  # 1.0 where Black stones
    current_white = (current == WHITE).to(torch.float32)  # 1.0 where White stones
    
    # Previous board (t-1)
    prev_black = (prev == BLACK).to(torch.float32)  # 1.0 where Black stones
    prev_white = (prev == WHITE).to(torch.float32)  # 1.0 where White stones
    
    # Previous-previous board (t-2)
    prev_prev_black = (prev_prev == BLACK).to(torch.float32)  # 1.0 where Black stones
    prev_prev_white = (prev_prev == WHITE).to(torch.float32)  # 1.0 where White stones
    
    # Stack current and previous states
    state = torch.stack([
        current_black, current_white,
        prev_black, prev_white,
        prev_prev_black, prev_prev_white
    ], dim=0)  # [6,19,19]
    return state


def play_self_play_game(
        policy_value_net: PolicyValueNet,
        device: torch.device,
        num_playouts: int,
        c_puct: float,
        temp_threshold: int = 8,
) -> List[Example]:
    """
    Play a full game via batched MCTS + the current policy_value_net.
    Returns a list of training examples (state, pi, z).

    `temp_threshold`: the move index t at which we switch from sampling (when t < temp_threshold)
    to picking argmax(π) (when t >= temp_threshold). This implements AlphaZero's "temperature" scheme.
    """
    examples: List[Example] = []
    BOARD_SIZE = policy_value_net.BOARD_SIZE  # 19 for full size
    game = GoGame(BOARD_SIZE)

    # End games of extraordinary length
    max_moves = 128 if BOARD_SIZE == 9 else 256

    move_count = 0
    while not game.game_over:
        # 1) Build state tensor
        state_tensor = state_to_tensor(game, device).unsqueeze(0)  # [1, 6,19,19]
        state_tensor = torch.concat([state_tensor,
                                     generate_influence_fields(state_tensor, sigma=1),  # Use full state for influence
                                     generate_influence_fields(state_tensor, sigma=3),
                                     generate_influence_fields(state_tensor, sigma=6)
                                     ], dim=1).squeeze()
        
        #State tensor shape: torch.Size([24, 9, 9])
        # print(f"State tensor shape: {state_tensor.shape}")
        # 2) Run batched MCTS to obtain visit counts
        root_game = game.clone()
        best_move, root = run_batched_mcts(root_game,
                                         policy_value_net,
                                         device,
                                         num_playouts,
                                         c_puct,
                                         training=True,
                                         temperature=1.0)
        
        # Get visit counts from root node
        pi = torch.zeros(BOARD_SIZE**2 + 1, dtype=torch.float32, device=device)  # +1 for pass
        total_N = 0
        for mv, child in root.children.items():
            if mv is None:
                pi[-1] = root.N[mv]  # Pass move is at the end
            else:
                idx = mv[0] * BOARD_SIZE + mv[1]
                pi[idx] = root.N[mv]
            total_N += root.N[mv]
        if total_N > 0:
            pi /= total_N

        # 3) Decide next action: sample or argmax depending on move_count
        if move_count < temp_threshold:
            # Sample from π with temperature 1.0
            pi_numpy = pi.detach().cpu().numpy()  # Move to CPU before converting to numpy
            legal_indices = pi_numpy.nonzero()[0]
            if legal_indices.size == 0:
                chosen_move = None  # must pass
            else:
                probs = pi_numpy[legal_indices]
                probs = probs / probs.sum()
                chosen_idx = random.choices(legal_indices.tolist(), weights=probs.tolist(), k=1)[0]
                if chosen_idx == BOARD_SIZE**2:  # Pass move
                    chosen_move = None
                else:
                    x, y = divmod(chosen_idx, BOARD_SIZE)
                    chosen_move = (x, y)
        else:
            # Deterministic: argmax
            top_idx = torch.argmax(pi).item()
            if pi[top_idx] == 0:
                chosen_move = None
            elif top_idx == BOARD_SIZE**2:  # Pass move
                chosen_move = None
            else:
                chosen_move = (top_idx // BOARD_SIZE, top_idx % BOARD_SIZE)

        # 4) Store example and play move
        examples.append((state_tensor, pi.clone(), None))

        if chosen_move is None:
            game.play_move()  # pass
        else:
            game.play_move(chosen_move[0], chosen_move[1])

        move_count += 1
        if move_count > max_moves:
            game.game_over = True

        # Clear some memory
        del state_tensor, pi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5) Game is over: compute final outcome z from Black's perspective
    score = game.score()
    b_score = score['black_score']
    w_score = score['white_score']
    if b_score > w_score:
        z = +1.0
    elif w_score > b_score:
        z = -1.0
    else:
        z = 0.0

    # 6) Fill in z for all stored examples
    finalized_examples: List[Example] = []
    for (state_tensor, pi_tensor, _) in examples:
        z_tensor = torch.tensor([z], dtype=torch.float32, device=device)
        finalized_examples.append((state_tensor, pi_tensor, z_tensor))

    # Clear memory
    del examples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return finalized_examples
