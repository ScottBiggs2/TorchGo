import torch
import numpy as np
import matplotlib.pyplot as plt

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.monte_carlo_tree_search_nodes import MCTSNode
from mcts.run_monte_carlo_tree_search import run_mcts
from training.self_play_system import ReplayBuffer, play_self_play_game
from training.training import train_policy_value_net
from play.human_vs_model import play_vs_net

def do_training():

    BOARD_SIZE = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(BOARD_SIZE).to(device)

    # Hyperparameters
    num_iterations = 8  # how many “generations” of self-play + training
    games_per_iteration = 16  # how many self-play games each generation
    num_playouts = 64  # MCTS playouts per move (tune to budget)
    c_puct = 0.6
    temp_threshold = 8 # layer < temp: check all policy draws, layer < temp, check c_puct proportion of draws
    replay_capacity = 4096
    batch_size = 64
    epochs_per_iter = 1
    lr = 1e-3
    l2_coef = 1e-4

    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    trained_net = train_policy_value_net(
        net=net,
        device=device,
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        num_playouts=num_playouts,
        c_puct=c_puct,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        epochs_per_iter=epochs_per_iter,
        lr=lr,
        l2_coef=l2_coef,
        classic_or_mini = True, # True = mini (9x9), False = classic (19x19)
    )

    torch.save(trained_net.state_dict(), "models/TorchGo-mini-light.pth")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    do_training()
