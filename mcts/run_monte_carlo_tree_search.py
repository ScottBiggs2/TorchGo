import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcts.monte_carlo_tree_search_nodes import MCTSNode
from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet


def run_mcts(root_game: GoGame,
             net: PolicyValueNet,
             device: torch.device,
             num_playouts: int,
             c_puct: float = 1.0) -> Optional[Tuple[int,int]]:
    """
    Run MCTS from the given root position. Repeat Selection/Expansion/Evaluation/Backprop `num_playouts` times.
    At the end, choose the root child with the highest visit count and return its move (x,y or None for pass).
    """
    root = MCTSNode(root_game.clone(), parent=None, move=None)

    for _ in range(num_playouts):
        node = root
        path = [node]

        # 1) SELECTION: descend until we find a node that is either:
        #    - not yet “expanded” (P is None), or
        #    - terminal (game_over = True)
        while True:
            if node.P is None or node.game.game_over:
                break
            # Node is already expanded and non‐terminal, so select its best child
            move, node = node.select_child(c_puct)
            path.append(node)

        # 2) EXPANSION & EVALUATION:
        if node.game.game_over:
            # Terminal node: directly compute the game outcome
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
            # Leaf node not yet expanded: call network
            value_leaf = node.expand_and_evaluate(net, device)

        # 3) BACKPROPAGATION:
        for i in range(len(path) - 1):
            parent_node = path[i]
            child_node = path[i + 1]
            move_taken = child_node.move

            parent_node.visits += 1
            parent_node.N[move_taken] += 1
            parent_node.W[move_taken] += value_leaf
            parent_node.Q[move_taken] = parent_node.W[move_taken] / parent_node.N[move_taken]

            # Flip value for the next step up (because players alternate)
            value_leaf = -value_leaf

        # Finally, increment visits at the leaf itself
        leaf_node = path[-1]
        leaf_node.visits += 1

    # After all playouts, pick the root’s move with highest visit count
    if root.P is None:
        # If root never expanded (no legal moves), must pass
        return None

    # Among each move in root.P, find root.N[move] (0 if missing). Pick argmax.
    best_move = max(root.P.keys(), key=lambda mv: root.N.get(mv, 0))
    return best_move