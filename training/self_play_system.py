from typing import Tuple, List, Optional
import torch
import random
import numpy as np
from collections import deque

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.monte_carlo_tree_search_nodes import MCTSNode

# Each example: (state_tensor, mcts_policy, z_value)
#   - state_tensor: torch.FloatTensor, shape [2,19,19]
#   - mcts_policy:  torch.FloatTensor, shape [361] (visit‐count distribution)
#   - z_value:      torch.FloatTensor, shape [1] (±1)
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

def state_to_tensor(game: GoGame, device: torch.device) -> torch.Tensor:
    """
    Convert the current position into a [2,19,19] float32 tensor:
      - channel 0: Black stones = 1.0, White stones = 0.0, Empty = 0.0
      - channel 1: White stones = 1.0, Black stones = 0.0, Empty = 0.0
    (Or alternatively use -1/1 encoding as you like. Just be consistent.)
    """
    board = game.board.float().to(device)  # shape [19,19], values ∈ {-1,0,+1}
    BLACK = game.BLACK
    WHITE = game.WHITE

    # We’ll put Black=1, White=1 on separate channels:
    black_plane = (board == BLACK).to(torch.float32)  # 1.0 where Black stones
    white_plane = (board == WHITE).to(torch.float32)  # 1.0 where White stones
    state = torch.stack([black_plane, white_plane], dim=0)  # [2,19,19]
    return state


def play_self_play_game(
        policy_value_net: PolicyValueNet,
        device: torch.device,
        num_playouts: int,
        c_puct: float,
        temp_threshold: int = 8,
        classic_or_mini: bool = True, # mini
) -> List[Example]:
    """
    Play a full game via MCTS + the current policy_value_net.
    Returns a list of training examples (state, pi, z).

    `temp_threshold`: the move index t at which we switch from sampling (when t < temp_threshold)
    to picking argmax(π) (when t >= temp_threshold). This implements AlphaZero’s “temperature” scheme.
    """
    examples: List[Example] = []
    BOARD_SIZE = policy_value_net.BOARD_SIZE  # 19 for full size
    game = GoGame(BOARD_SIZE)

    # End games of extrordinary length
    if classic_or_mini == True:  # if mini or 9x9 board
        komi = 0.5 # mostly just to break ties
        max_moves = 128
    else:
        komi = 6.5 # traditional compensation for white playing 2nd, usually between 5.5 and 7.5
        max_moves = 512

    move_count = 0
    while not game.game_over:
        # 1) Build state tensor
        state_tensor = state_to_tensor(game, device)  # [2,19,19]

        # 2) Run MCTS to obtain visit counts
        #    We need not return the “best move” here; we want the full distribution π.
        #    To do that, slightly modify run_mcts to return the root node itself.
        root = MCTSNode(game.clone(), parent=None, move=None)
        for _ in range(num_playouts):
            node = root
            path = [node]
            # Selection
            while True:
                if node.P is None or node.game.game_over:
                    break
                mv, node = node.select_child(c_puct)
                path.append(node)
            # Expansion & Evaluation
            if node.game.game_over:
                terr = node.game.estimate_territory()
                b_ter = terr['black_territory']
                w_ter = terr['white_territory']
                if b_ter > w_ter:
                    value_leaf = +1.0
                elif w_ter > b_ter:
                    value_leaf = -1.0
                else:
                    value_leaf = 0.0
            else:
                value_leaf = node.expand_and_evaluate(policy_value_net, device)
            # Backpropagation
            for i in range(len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                mv_taken = child.move
                parent.visits += 1
                parent.N[mv_taken] += 1
                parent.W[mv_taken] += value_leaf
                parent.Q[mv_taken] = parent.W[mv_taken] / parent.N[mv_taken]
                value_leaf = -value_leaf
            path[-1].visits += 1

        # 3) Extract π: normalize visits at root over all legal moves
        pi = torch.zeros( (BOARD_SIZE**2), dtype=torch.float32, device=device)
        total_N = 0
        for mv, child in root.children.items():
            if mv is None:
                continue
            idx = mv[0] * BOARD_SIZE + mv[1]
            pi[idx] = root.N[mv]
            total_N += root.N[mv]
        if total_N > 0:
            pi /= total_N

        # 4) Decide next action: sample or argmax depending on move_count
        if move_count < temp_threshold:
            # Sample from π with temperature 1.0 (i.e. directly proportional)
            # pi_numpy = pi.cpu().numpy()

            pi_numpy = pi.detach().numpy(force = True)
            legal_indices = pi_numpy.nonzero()[0]
            if legal_indices.size == 0:
                chosen_move = None  # must pass
            else:
                probs = pi_numpy[legal_indices]
                probs = probs / probs.sum()
                chosen_idx = random.choices(legal_indices.tolist(), weights=probs.tolist(), k=1)[0]
                x, y = divmod(chosen_idx, BOARD_SIZE)
                chosen_move = (x, y)
        else:
            # Deterministic: argmax
            top_idx = torch.argmax(pi).item()
            if pi[top_idx] == 0:
                chosen_move = None
            else:
                chosen_move = (top_idx // BOARD_SIZE, top_idx % BOARD_SIZE)

        # 5) Compute z later, but for now store (state, pi) and placeholder for z
        examples.append((state_tensor, pi.clone(), None))

        # 6) Play the move in the real game
        if chosen_move is None:
            game.play_move()  # pass
        else:
            game.play_move(chosen_move[0], chosen_move[1])

        move_count += 1

        if move_count > max_moves:
            game.game_over = True

    # 7) Game is over: compute final outcome z from Black’s perspective
    terr = game.estimate_territory()
    b_ter = terr['black_territory']
    w_ter = terr['white_territory'] + komi
    if b_ter > w_ter:
        z = +1.0
    elif w_ter > b_ter:
        z = -1.0
    else:
        z = 0.0

    # 8) Fill in z for all stored examples
    finalized_examples: List[Example] = []
    for (state_tensor, pi_tensor, _) in examples:
        z_tensor = torch.tensor([z], dtype=torch.float32, device=device)
        finalized_examples.append((state_tensor, pi_tensor, z_tensor))

    return finalized_examples
