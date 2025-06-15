import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

import time
import numpy as np
from typing import Tuple, List, Optional

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.monte_carlo_tree_search_nodes import MCTSNode
from training.self_play_system import ReplayBuffer, play_self_play_game

Example = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# ------------------------------
# 3.1 A PyTorch Dataset for our examples
# ------------------------------
class GameDataset(Dataset):
    def __init__(self, examples: List[Example], transforms = None):
        self.examples = examples
        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        # state, pi, z = self.examples[idx]
        sample = self.examples[idx]

        # Let's hope this doesn't nuke everything?
        if self.transforms:
            sample = self.transforms(sample)

        state, pi, z = sample

        return state, pi, z

def compute_loss(
    policy_pred: torch.Tensor,  # [B,361], *softmax* probabilities
    value_pred: torch.Tensor,   # [B,1], tanh outputs in [-1,+1]
    pi_target: torch.Tensor,    # [B,361], MCTS visit‐count distribution (sum=1)
    z_target: torch.Tensor,     # [B,1], exact ±1 labels from final outcome
    net: PolicyValueNet,
    l2_coef: float
) -> torch.Tensor:
    """
    policy_pred:  [B,361] = softmax output
    value_pred:   [B,1]   = tanh output in [-1,+1]
    pi_target:    [B,361] = MCTS target probabilities (sum = 1)
    z_target:     [B,1]   = ±1 final game outcomes (Black=+1, White=−1)

    Loss = (value_loss) + (policy_loss) + (l2_coeff * ||θ||²)

    - value_loss = MSE(value_pred, z_target)
    - policy_loss = –∑(π_target · log π_pred) averaged over batch
    - l2_loss = l2_coef * ∑‖param‖₂²
    """

    # 1) Value loss: MSE between tanh‐output in [-1,+1] and z_target ∈ {–1,+1}
    #    If you instead want BCE, you would change value_pred→sigmoid and z_target→{0,1}.
    value_pred_flat = value_pred.view(-1)       # [B]
    z_flat = z_target.view(-1)                  # [B]
    value_loss = F.mse_loss(value_pred_flat, z_flat)

    # 2) Policy loss: cross‐entropy between target π and predicted π
    #    policy_pred is *softmax* over logits, so we take log once here:
    logp = torch.log(policy_pred + 1e-8)        # [B,361]
    policy_loss = - (pi_target * logp).sum(dim=1).mean()  # scalar

    # 3) L2 regularization on all parameters
    l2_loss = torch.tensor(0.0, device=policy_pred.device)
    for param in net.parameters():
        l2_loss = l2_loss + torch.norm(param, p=2) ** 2
    l2_loss = l2_coef * l2_loss

    return value_loss + policy_loss + l2_loss

class GoBoardTransform:
    """
    Custom transform for Go board data augmentation.
    Handles both state tensor [B,16,19,19] and policy tensor [B,361].
    The policy tensor includes an extra element at the end for passing.
    """
    def __init__(self, p_horizontal: float = 0.5, p_vertical: float = 0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical

    def __call__(self, sample: Example) -> Example:
        state, pi, z = sample
        
        # Get board size from state tensor (last two dimensions are equal)
        BOARD_SIZE = state.shape[-1]  # 19 for classic, 9 for mini

        # Randomly decide whether to flip
        flip_h = np.random.random() < self.p_horizontal
        flip_v = np.random.random() < self.p_vertical

        if not (flip_h or flip_v):
            return sample

        # 1) Transform state tensor [B,16,19,19]
        if flip_h:
            state = torch.flip(state, dims=[-1])  # Flip last dimension (horizontal)
        if flip_v:
            state = torch.flip(state, dims=[-2])  # Flip second-to-last dimension (vertical)

        # 2) Transform policy tensor [B,board_size**2+1]
        if flip_h or flip_v:
            # Split policy into board moves and pass move
            pi_board = pi[..., :-1]  # All but last element (board moves)
            pi_pass = pi[..., -1:].unsqueeze(0)   # Last element (pass move)
            
            # Reshape board moves to 2D
            pi_2d = pi_board.view(1, BOARD_SIZE, BOARD_SIZE)
            
            if flip_h:
                pi_2d = torch.flip(pi_2d, dims=[-1])
            if flip_v:
                pi_2d = torch.flip(pi_2d, dims=[-2])
            
            # Reshape back to 1D and concatenate with pass move
            pi_board = pi_2d.view(1, BOARD_SIZE * BOARD_SIZE)
            # print(f"pi_board shape: {pi_board.shape}, pi_pass shape: {pi_pass.shape}")
            pi = torch.cat([pi_board, pi_pass], dim=1)

        # 3) Value z remains unchanged as it's game outcome
        return state, pi, z

def train_policy_value_net(
    net: PolicyValueNet,
    device: torch.device,
    num_iterations: int,
    games_per_iteration: int,
    num_playouts: int,
    c_puct: float,
    temp_threshold: int,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    epochs_per_iter: int,
    lr: float,
    l2_coef: float,
    classic_or_mini: bool, # True = mini (9x9), False = classic (19x19)
):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay= 1e-4)

    # Create dataset with Go board augmentation
    transform = GoBoardTransform(p_horizontal=0.5, p_vertical=0.5)

    iter_start = time.time()
    for it in range(num_iterations):
        print(f"\n=== Iteration {it+1}/{num_iterations} ===")
        # 1) Generate self-play games and collect examples
        iteration_examples: List[Example] = []
        game_start_time = time.time()

        for g in range(games_per_iteration):
            examples = play_self_play_game(net, device, num_playouts, c_puct, temp_threshold, classic_or_mini)
            iteration_examples.extend(examples)
            print(f"  Self-play game {g+1}/{games_per_iteration}: {len(examples)} positions.")
        game_end_time = time.time()

        # 2) Push examples into replay buffer
        replay_buffer.push(iteration_examples)
        print(f"  Replay buffer size: {len(replay_buffer)}/{replay_buffer.capacity}")

        # 3) Train for a few epochs on random mini-batches from the buffer
        if len(replay_buffer) < batch_size:
            print("  Not enough samples in buffer yet to train.")
            continue

        # Create a DataLoader over a snapshot of buffer’s contents (to avoid sampling anew each epoch)
        # by randomly flipping on x and y axes we access all 4 axes of symmetry in Go
        data_snapshot = list(replay_buffer.buffer)
        dataset = GameDataset(data_snapshot, transforms = None) # transform not quite working
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        states, pi_targets, z_targets = next(iter(loader))

        net.train()
        epoch_start = time.time()
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            for states, pi_targets, z_targets in loader:
                # states: [B,4,19,19], pi_targets: [B,361], z_targets: [B,1]
                states = states.to(device)
                pi_targets = torch.nan_to_num(pi_targets).to(device) + 1e-8
                z_targets = z_targets.to(device)

                optimizer.zero_grad()
                policy_out, value_out = net(states)      # policy_out: [B,361], value_out: [B,1]
                loss = compute_loss(policy_out, value_out, pi_targets, z_targets, net, l2_coef= 1e-2)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"    Epoch {epoch+1}/{epochs_per_iter}: Loss = {epoch_loss/len(loader):.4f}")
        epoch_end = time.time()
        iter_end = time.time()
        print(f" === \n Iteration {it+1} took: {(iter_end - iter_start):.2f}s\n Games {(game_end_time - game_start_time)/games_per_iteration:.2f}s per game \n Epochs avg {(epoch_end - epoch_start)/epochs_per_iter:.2f}s per epoch \n ===")
    return net

