import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.monte_carlo_tree_search_nodes import MCTSNode
from training.self_play_system import state_to_tensor
from play.human_vs_model import plot_board, plot_policy, get_user_move

def review_game(
        policy_value_net: PolicyValueNet,
        device: torch.device,
        top_k: int,
        board_size: int,
        return_moves=True,
    ):
    """
    Interactive review mode: user inputs moves one by one (or 'pass'), engine shows evaluation and suggestions.
    Tracks value estimates and territory estimates over the course of the game and plots them at the end.
    """
    game = GoGame(board_size)
    move_history = []
    value_history = []
    territory_history = []

    BOARD_SIZE = game.BOARD_SIZE
    BLACK = game.BLACK
    WHITE = game.WHITE

    NUM_MOVES = BOARD_SIZE * BOARD_SIZE
    print("Enter moves to review. Format: 'row col' (0-based). Type 'pass' to pass, 'done' to end review.")

    while True:
        # Display current board
        fig = plot_board(game)
        display(fig)

        # Get network evaluation at current position
        state_tensor = state_to_tensor(game, device).unsqueeze(0)  # [1,2,BOARD_SIZE,BOARD_SIZE]
        with torch.no_grad():
            raw_policy, value = policy_value_net(state_tensor)
        raw_policy = raw_policy.squeeze(0)  # [NUM_MOVES]

        print("Network showing raw policy prior...\n")
        fig_raw = plot_policy(game, raw_policy)
        display(fig_raw)
        value = float(value.item())  # scalar in [-1,1]

        # Show network's top-3 suggestions
        probs = raw_policy.cpu().numpy()
        # Zero out illegal moves
        legal_mask = np.zeros(NUM_MOVES, dtype=bool)
        for idx in range(NUM_MOVES):
            x, y = divmod(idx, BOARD_SIZE)
            if game.is_legal(x, y):
                legal_mask[idx] = True
        probs_masked = probs * legal_mask
        if probs_masked.sum() > 0:
            probs_norm = probs_masked / probs_masked.sum()
        else:
            probs_norm = legal_mask.astype(float) / legal_mask.sum()
        top_indices = probs_norm.argsort()[-top_k:][::-1]
        suggestions = [(idx // BOARD_SIZE, idx % BOARD_SIZE, probs_norm[idx]) for idx in top_indices]

        print(f"Evaluation (value ∈ [-1,+1], +1=Black wins, -1=White wins): {value:.3f}")
        print(f"Top-{top_k} suggestions (row, col, probability):")
        for (rx, ry, p) in suggestions:
            print(f"  → ({rx}, {ry}): {p:.3f}")

        # Record history
        move_history.append(game.clone())  # store a snapshot
        value_history.append(value)
        terr = game.estimate_territory()
        territory_history.append((terr['black_territory'], terr['white_territory']))

        # Prompt user for next move
        move_str = input("Next move ('row col', 'pass', or 'done'): ").strip().lower()
        if move_str == 'done':
            break
        if move_str == 'pass':
            game.play_move()
            continue
        parts = move_str.split()
        if len(parts) != 2:
            print("Invalid input format. Please try again.")
            continue
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("Invalid numbers. Please try again.")
            continue
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            print(f"Coordinates must be between 0 and {BOARD_SIZE-1}.")
            continue
        if not game.is_legal(x, y):
            print(f"Move ({x}, {y}) is illegal. Try again.")
            continue
        game.play_move(x, y)

    # Game is over: show final board and result
    fig_final = plot_board(game)
    display(fig_final)

    terr = game.estimate_territory()
    b_ter = terr['black_territory']
    w_ter = terr['white_territory']
    print(f"Final score → Black: {b_ter}, White: {w_ter}")
    if b_ter > w_ter:
        print("Black wins!")
        if human_plays_black:
            print("You win!")
        else:
            print("Opponent wins!")
    elif w_ter > b_ter:
        print("White wins!")
        if not human_plays_black:
            print("You win!")
        else:
            print("Opponent wins!")
    else:
        print("It's a tie!")

    if return_moves:
        print(f"Move recording: \n")
        game.print_move_log

    # After review ends, plot evaluation and territory over move number
    moves = list(range(len(value_history)))
    black_ter, white_ter = zip(*territory_history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(moves, value_history, marker='o')
    ax1.set_title("Value (Win Estimate) over Moves")
    ax1.set_xlabel("Move Number")
    ax1.set_ylabel("Value (–1 to +1)")
    ax1.grid(True)

    ax2.plot(moves, black_ter, label="Black Territory")
    ax2.plot(moves, white_ter, label="White Territory")
    ax2.set_title("Estimated Territory over Moves")
    ax2.set_xlabel("Move Number")
    ax2.set_ylabel("Territory Count")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.close(fig)
    return fig

