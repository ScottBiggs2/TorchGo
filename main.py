import torch
import numpy as np
import matplotlib.pyplot as plt
import gc  # Add garbage collector
import os
import psutil  # For memory monitoring

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from models.policy_value_transformer import PolicyValueTransformer   
# from mcts.monte_carlo_tree_search_nodes import MCTSNode
# from mcts.run_monte_carlo_tree_search import run_mcts
from training.self_play_system import ReplayBuffer, play_self_play_game
from training.training import train_policy_value_net, count_parameters
from play.human_vs_model import play_vs_net

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def __main__():
    # Clear any existing CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection
    
    BOARD_SIZE = 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueTransformer(BOARD_SIZE, d_model = 128,
                                nhead = 4, num_layers = 6,
                                num_head_layers = 2, dropout = 0.1)
    
    model_path = "models/TorchGo-transformer-9x9.pth"
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    print(f"Number of parameters: {count_parameters(net)}")

    # Games trained on (as of 18/06/2025): 10 + 5x10 + 36x10 = 420
    # Epochs trained on (as of 18/06/2025): 3 + 5x3 + 36x3 = 136

    # Num Playouts | ~% of ideal results : Q = 1 -1/sqrt(N) or proportional to log(N) | time per game
    # 128 | 92%  |  5 minutes/game  |
    # 64  | 85%  |  1 minutes/game  |
    # 800 | 96%  | * This is the setting used by AlphaGo for 9x9       

    # Model params notes:
    # Update to nheads = 1 to get stronger long range depenency capture 
    # increase model dim (min_x where 2^x >= board_size**2 - naieve but at least rule based)
    
    # hyperparams notes:
    # strength ~= f(num_playouts) ( see above ^^^)
    # temp_threshold = 4 - increase to widen search. I have a hunch that early in training width > depth for PPO
    
    # to-do:
    # make epochs_per_iter a function of proportion of replay capacity filled (maybe also batch-size?)
    # increase sparsity penalty
    # add cosine annealing or reduce on plateau LR scheduler
    # fix the colab notebook so people can actually see it? I hate colab :(
    
    # Redesign: learn a likely-to-be controlled territory mask function to help value head converge

    # Hyperparameters
    num_iterations = 50  
    games_per_iteration = 10  
    num_playouts = 64  
    c_puct = 1.5
    temp_threshold = 4
    replay_capacity = 128000
    batch_size = 256
    epochs_per_iter = 3  
    lr = 1e-3
    l2_coef = 1e-4

    try:
        print("Initial memory state:")
        print_memory_usage()
        
        replay_buffer = ReplayBuffer(capacity=replay_capacity)

        trained_net = train_policy_value_net(
            net=net,
            device=device,
            num_iterations=num_iterations,
            games_per_iteration=games_per_iteration,
            num_playouts=num_playouts,
            c_puct=c_puct,
            temp_threshold=temp_threshold,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            epochs_per_iter=epochs_per_iter,
            lr=lr,
            l2_coef=l2_coef,
            model_save_path = model_path
        )

        # Save the model
        # torch.save(trained_net.state_dict(), "models/TorchGo-transformer-9x9.pth")
        
        # Clean up
        del trained_net, replay_buffer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("\nFinal memory state:")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Starting main.py")
    __main__()

