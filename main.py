# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
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

def __main__():

    BOARD_SIZE = 9

    # 1) Create a Game and a PolicyValueNet
    # game = GoGame(BOARD_SIZE)
    # net = PolicyValueNet(BOARD_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(BOARD_SIZE).to(device)
    # Optionally, load a checkpoint:
    # net.load_state_dict(torch.load("policy_value_net_final.pth", map_location=device))

    # play_vs_net(net, device, num_playouts=32, c_puct=0.8, board_size = BOARD_SIZE)

    # # Hyperparameters
    num_iterations = 5             # how many “generations” of self-play + training
    games_per_iteration = 2        # how many self-play games each generation
    num_playouts = 16              # MCTS playouts per move (tune to budget)
    c_puct = 0.6
    replay_capacity = 50
    batch_size = 4
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
        l2_coef=l2_coef
    )

    # # Save the final model
    torch.save(trained_net.state_dict(), "models/test.pth")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    __main__()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

