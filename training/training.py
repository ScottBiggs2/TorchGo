import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        state, pi, z = self.examples[idx]
        return state, pi, z

# ------------------------------
# 3.2 Loss Functions
# ------------------------------
def compute_loss(
    policy_pred: torch.Tensor,  # [B,361], log-probabilities after log()
    value_pred: torch.Tensor,   # [B,1]
    pi_target: torch.Tensor,    # [B,361], probabilities from MCTS
    z_target: torch.Tensor,     # [B,1], ±1 outcomes
    net: PolicyValueNet,
    l2_coef: float
) -> torch.Tensor:
    """
    - policy_pred should be the log‐probabilities (i.e. log(policy) from net forward).
    - pi_target is the MCTS‐derived target distribution.
    - value_pred is the tanh output. z_target is ±1 real outcome.
    Loss = λ1 * (z - v)^2  +  λ2 * ( - sum(pi_target * log(policy_pred)) )  +  λ3 * sum(W^2) over all weights.
    Here we’ll just take λ1=1, λ2=1, λ3=l2_coef.
    """
    # 1) Value loss: MSE
    value_loss = F.mse_loss(value_pred.view(-1), z_target.view(-1))

    # 2) Policy loss: cross‐entropy between π_target and predicted policy
    #    policy_pred is log(prob) for numerical stability
    policy_loss = -(pi_target * policy_pred).sum(dim=1).mean()

    # 3) L2 regularization
    l2_loss = torch.tensor(0.0, device=policy_pred.device)
    for param in net.parameters():
        l2_loss += torch.norm(param, p=2) ** 2
    l2_loss = l2_coef * l2_loss

    return value_loss + policy_loss + l2_loss

# ------------------------------
# 3.3 Training Configuration & Main Loop
# ------------------------------
def train_policy_value_net(
    net: PolicyValueNet,
    device: torch.device,
    num_iterations: int,
    games_per_iteration: int,
    num_playouts: int,
    c_puct: float,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    epochs_per_iter: int,
    lr: float,
    l2_coef: float
):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)

    for it in range(num_iterations):
        print(f"\n=== Iteration {it+1}/{num_iterations} ===")
        # 1) Generate self-play games and collect examples
        iteration_examples: List[Example] = []
        for g in range(games_per_iteration):
            examples = play_self_play_game(net, device, num_playouts, c_puct)
            iteration_examples.extend(examples)
            print(f"  Self-play game {g+1}/{games_per_iteration}: {len(examples)} positions.")

        # 2) Push examples into replay buffer
        replay_buffer.push(iteration_examples)
        print(f"  Replay buffer size: {len(replay_buffer)}/{replay_buffer.capacity}")

        # 3) Train for a few epochs on random mini-batches from the buffer
        if len(replay_buffer) < batch_size:
            print("  Not enough samples in buffer yet to train.")
            continue

        # Create a DataLoader over a snapshot of buffer’s contents (to avoid sampling anew each epoch)
        data_snapshot = list(replay_buffer.buffer)
        dataset = GameDataset(data_snapshot)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        net.train()
        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            for states, pi_targets, z_targets in loader:
                # states: [B,2,19,19], pi_targets: [B,361], z_targets: [B,1]
                states = states.to(device)
                pi_targets = pi_targets.to(device)
                z_targets = z_targets.to(device)

                optimizer.zero_grad()
                policy_out, value_out = net(states)      # policy_out: [B,361], value_out: [B,1]
                log_policy = torch.log(policy_out + 1e-8)  # avoid log(0)
                loss = compute_loss(log_policy, value_out, pi_targets, z_targets, net, l2_coef)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            print(f"    Epoch {epoch+1}/{epochs_per_iter}: Loss = {epoch_loss/len(loader):.4f}")

        # Model Saving - not yet implemented
        # if (it+1)%save_every == 0 or (it+1) == num_iterations:
        #     torch.save(net.state_dict(), "models/saved_models/policy_value_net_it_{it+1}.pth")

        # 4) (Optional) Evaluate net against a fixed opponent or older snapshot
        # For example, let net play 20 games vs. the previous iteration’s net and measure win-rate.

    # After all iterations, return the trained net
    return net
