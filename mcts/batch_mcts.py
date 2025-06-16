import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.batch_mcts_node import MCTSNode, generate_influence_fields

class BatchedMCTS:
    def __init__(self, net: PolicyValueNet, device: torch.device, batch_size=16, 
                 c_puct=1.0, training=False, dirichlet_alpha=0.03):
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.training = training
        self.dirichlet_alpha = dirichlet_alpha
        self.queue = []

    def _backpropagate(self, path: List[MCTSNode], leaf_value: float):
        """Backpropagate value with perspective flipping"""
        value = leaf_value
        print(f"DEBUG: Starting backprop with leaf value: {value}")
        
        # Traverse backwards from leaf to root
        for i in range(len(path) - 1, -1, -1):  # Include leaf node
            node = path[i]
            
            # Update node's own visits
            node.visits += 1
            
            # If not the leaf node, update move statistics
            if i < len(path) - 1:
                child = path[i + 1]
                move = child.move
                
                # Revert virtual loss first
                node.revert_virtual_loss(move)
                
                # Update statistics for this move
                node.N[move] += 1
                node.W[move] += value
                node.Q[move] = node.W[move] / node.N[move]
                
                print(f"DEBUG: Node {i} - Move: {move}, Visits: {node.visits}, N[move]: {node.N[move]}, W[move]: {node.W[move]}, Q[move]: {node.Q[move]}")
            
            # Flip value for parent's perspective
            value = -value

    def add_to_queue(self, node: MCTSNode, path: List[MCTSNode]):
        self.queue.append((node, path))
        if len(self.queue) >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        if not self.queue:
            return
            
        print(f"\nDEBUG: Processing batch of {len(self.queue)} nodes")
        
        # Filter out terminal nodes
        terminal_nodes = []
        non_terminal_nodes = []
        non_terminal_paths = []
        
        for (node, path) in self.queue:
            if node.game.game_over:
                terminal_nodes.append((node, path))
            else:
                non_terminal_nodes.append(node)
                non_terminal_paths.append(path)
        
        # Handle terminal nodes first
        for node, path in terminal_nodes:
            score = node.game.score()
            outcome = 1.0 if score['black_score'] > score['white_score'] else -1.0
            if node.game.current_player == node.game.WHITE:
                outcome = -outcome
            self._backpropagate(path, outcome)
        
        # Process non-terminal nodes
        if non_terminal_nodes:
            state_tensors = [node.get_state_tensor(self.device) for node in non_terminal_nodes]
            batch = torch.cat(state_tensors, dim=0)
            
            print(f"DEBUG: Batch shape: {batch.shape}")
            print(f"DEBUG: Batch values range: [{batch.min()}, {batch.max()}]")
            
            with torch.no_grad():
                policy_logits_batch, values_batch = self.net(batch)
            policy_logits_batch = policy_logits_batch.cpu()
            values_batch = values_batch.squeeze(1).cpu().numpy()
            
            print(f"DEBUG: Policy logits shape: {policy_logits_batch.shape}")
            print(f"DEBUG: Policy logits range: [{policy_logits_batch.min()}, {policy_logits_batch.max()}]")
            print(f"DEBUG: Values shape: {values_batch.shape}")
            print(f"DEBUG: Values range: [{values_batch.min()}, {values_batch.max()}]")
            
            for (node, path), policy_logits, value in zip(
                zip(non_terminal_nodes, non_terminal_paths), policy_logits_batch, values_batch):
                print(f"DEBUG: Expanding node with value {value}")
                # Ensure we're expanding the node with the correct policy logits
                if node.P is None:  # Double check node hasn't been expanded
                    self._expand_node(node, policy_logits)
                    # Verify expansion was successful
                    if node.P is None:
                        print("ERROR: Node expansion failed!")
                        continue
                self._backpropagate(path, float(value))
        
        self.queue.clear()

    def _expand_node(self, node: MCTSNode, policy_logits: torch.Tensor):
        board_size = node.game.BOARD_SIZE
        total_moves = board_size**2 + 1
        
        print(f"\nDEBUG: Expanding node with policy logits shape: {policy_logits.shape}")
        print(f"DEBUG: Policy logits values: {policy_logits}")
        
        # Ensure policy_logits is 2D [batch=1, moves]
        if policy_logits.dim() == 1:
            policy_logits = policy_logits.unsqueeze(0)
        
        # Create legal moves mask [moves]
        legal_mask = torch.zeros(total_moves, dtype=torch.bool)
        legal_moves = []
        for idx in range(board_size**2):
            x, y = divmod(idx, board_size)
            if node.game.is_legal(x, y):
                legal_mask[idx] = True
                legal_moves.append((x, y))
        legal_mask[-1] = True  # Pass is always legal
        legal_moves.append(None)  # Add pass move
        
        print(f"DEBUG: Found {len(legal_moves)} legal moves")
        print(f"DEBUG: Legal mask: {legal_mask}")
        
        # Apply mask and softmax
        masked_logits = policy_logits.clone()  # [1, moves]
        masked_logits[0, ~legal_mask] = -float('inf')  # Apply mask to first (and only) batch
        print(f"DEBUG: Masked logits: {masked_logits}")
        
        # Apply softmax to get probabilities
        legal_probs = F.softmax(masked_logits, dim=1)  # [1, moves]
        print(f"DEBUG: Legal probabilities after softmax: {legal_probs}")
        
        # Store priors
        priors = {}
        legal_indices = legal_mask.nonzero(as_tuple=True)[0]
        for idx in legal_indices:
            prob = legal_probs[0, idx].item()  # Get probability for this move
            if idx == board_size**2:
                priors[None] = prob  # Pass move
            else:
                x, y = divmod(idx.item(), board_size)
                priors[(x, y)] = prob
        
        print(f"DEBUG: Priors before Dirichlet: {priors}")
        
        # Add Dirichlet noise during training at root
        if self.training and node.parent is None and priors:
            dirichlet = torch.distributions.dirichlet.Dirichlet(
                torch.ones(len(priors)) * self.dirichlet_alpha
            )
            noise = dirichlet.sample()
            moves = list(priors.keys())
            for i, move in enumerate(moves):
                priors[move] = 0.75 * priors[move] + 0.25 * noise[i]
            print(f"DEBUG: Priors after Dirichlet: {priors}")
        
        # Ensure no zero probabilities
        min_prob = 1e-8
        for move in priors:
            if priors[move] < min_prob:
                priors[move] = min_prob
        
        # Normalize probabilities
        total_prob = sum(priors.values())
        for move in priors:
            priors[move] /= total_prob
        
        print(f"DEBUG: Final normalized priors: {priors}")
        
        # Set the node's priors and initialize statistics
        node.P = priors
        for move in priors:
            node.N[move] = 0
            node.W[move] = 0.0
            node.Q[move] = 0.0
        
        # Verify expansion was successful
        if node.P is None:
            print("ERROR: Node expansion failed to set priors!")
