import math
from typing import Optional, Tuple, List
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.batch_mcts_node import MCTSNode, generate_influence_fields
from mcts.batch_mcts import BatchedMCTS

def run_batched_mcts(root_game: GoGame, 
                     net: PolicyValueNet, 
                     device: torch.device, 
                     num_playouts: int,
                     c_puct: float = 1.0,
                     temperature: float = 1.0,
                     training: bool = False) -> Tuple[Optional[Tuple[int, int]], MCTSNode]:
    """
    Run batched MCTS with configurable exploration parameters.
    
    Args:
        root_game: Initial game state
        net: Policy-value network
        device: Torch device
        num_playouts: Number of simulations
        c_puct: Exploration constant
        temperature: Controls move selection randomness (1.0=exploration, 0.0=greedy)
        training: Whether in training mode (affects Dirichlet noise)
        
    Returns:
        Tuple of (best move (x,y) or None for pass, root node)
    """
    root = MCTSNode(root_game.clone())
    batched_mcts = BatchedMCTS(
        net, device, 
        batch_size=16, 
        c_puct=c_puct,
        training=training,
        dirichlet_alpha=0.03
    )
    
    # Run simulations
    for sim in range(num_playouts):
        # print(f"\nDEBUG: Starting simulation {sim + 1}/{num_playouts}")
        node = root
        path = [node]
        
        # Selection phase
        while True:
            if node.game.game_over:
                # Terminal node - calculate outcome
                score = node.game.score()
                outcome = 1.0 if score['black_score'] > score['white_score'] else -1.0
                if node.game.current_player == node.game.WHITE:
                    outcome = -outcome
                # print(f"DEBUG: Terminal node reached with outcome {outcome}")
                batched_mcts._backpropagate(path, outcome)
                break
                
            if node.P is None:
                # print(f"DEBUG: Node needs expansion, adding to batch queue")
                batched_mcts.add_to_queue(node, path)
                # Only process batch if it's full or this is the last simulation
                if len(batched_mcts.queue) >= batched_mcts.batch_size or sim == num_playouts - 1:
                    batched_mcts.process_batch()
                    # If the node still isn't expanded, something is wrong
                    if node.P is None:
                        print("ERROR: Node failed to expand after batch processing")
                        break
                else:
                    # Skip to next simulation if we're waiting for batch to fill
                    break
                # Now we can continue with the expanded node
                continue
                
            # Check if all children are terminal states
            all_children_terminal = True
            for move in node.P:
                if move not in node.children:
                    all_children_terminal = False
                    break
                if not node.children[move].game.game_over:
                    all_children_terminal = False
                    break
            
            if all_children_terminal:
                # All children are terminal, evaluate this node
                score = node.game.score()
                outcome = 1.0 if score['black_score'] > score['white_score'] else -1.0
                if node.game.current_player == node.game.WHITE:
                    outcome = -outcome
                # print(f"DEBUG: All children terminal, outcome {outcome}")
                batched_mcts._backpropagate(path, outcome)
                break
                
            move, child = node.select_child(batched_mcts.c_puct)
            # print(f"DEBUG: Selected move {move} with UCB score {child.Q.get(move, 0)}")
            path.append(child)
            node = child
    
    # Process any remaining in the batch queue
    # print("\nDEBUG: Processing final batch")
    batched_mcts.process_batch()
    
    # Select move based on visit counts
    if not root.P:
        # print("DEBUG: No legal moves (root.P is empty)")
        return None, root  # No legal moves
    
    visit_counts = {move: root.N[move] for move in root.P}
    total_visits = sum(visit_counts.values())
    # print(f"DEBUG: Total visits: {total_visits}")
    # print(f"DEBUG: Visit counts: {visit_counts}")
    
    if temperature < 0.01 or not training:
        # Greedy selection
        best_move = max(visit_counts, key=visit_counts.get)
        # print(f"DEBUG: Greedy selection chose move {best_move} with {visit_counts[best_move]} visits")
    else:
        # Ensure minimum visit count to avoid zero weights
        min_visits = 1
        visit_counts = {move: max(count, min_visits) for move, count in visit_counts.items()}
        
        # Apply temperature scaling
        scaled_visits = {
            move: (count ** (1/temperature)) 
            for move, count in visit_counts.items()
        }
        total_scaled = sum(scaled_visits.values())
        # print(f"DEBUG: Total scaled visits: {total_scaled}")
        # print(f"DEBUG: Scaled visits: {scaled_visits}")
        
        # Calculate probabilities with minimum weight
        min_weight = 1e-8
        probs = {move: max(count / total_scaled, min_weight) for move, count in scaled_visits.items()}
        
        # Normalize probabilities
        total_prob = sum(probs.values())
        probs = {move: prob / total_prob for move, prob in probs.items()}
        
        # print(f"DEBUG: Final probabilities: {probs}")
        # print(f"DEBUG: Sum of probabilities: {sum(probs.values())}")
        
        # Sample from probability distribution
        moves, weights = zip(*probs.items())
        best_move = random.choices(moves, weights=weights, k=1)[0]
        # print(f"DEBUG: Sampled move {best_move} with probability {probs[best_move]}")
    
    return best_move, root