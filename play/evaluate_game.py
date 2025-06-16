import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.run_batched_mcts import run_batched_mcts
from training.self_play_system import state_to_tensor, generate_influence_fields
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


    while not game.game_over:
        # Plot current board
        fig = plot_board(game)
        display(fig)

        # Get user's move
        user_move = get_user_move(game)
        if user_move is None:
            game.play_move()  # pass
            print("You passed.\n")
        elif user_move is True:
            game.game_over = True
            print(f"You ended the game.")
        else:
            game.play_move(user_move[0], user_move[1])
            print(f"You played at {user_move}.\n")

        # Get network's evaluation
        state_tensor = state_to_tensor(game, device).unsqueeze(0)  # [1,6,19,19]
        state_tensor = torch.concat([state_tensor,
                                   generate_influence_fields(state_tensor, sigma=1),
                                   generate_influence_fields(state_tensor, sigma=3),
                                   generate_influence_fields(state_tensor, sigma=6)
                                  ], dim=1)

        with torch.no_grad():
            policy, value = policy_value_net(state_tensor)
            raw_policy = policy.squeeze(0)  # [NUM_MOVES]
            value = float(value.item())

        print("Network showing raw policy prior...\n")
        fig_raw = plot_policy(game, raw_policy)
        display(fig_raw)
        
        # Run batched MCTS to get move suggestions
        best_move, root = run_batched_mcts(
            game.clone(),
            policy_value_net,
            device,
            num_playouts=100,  # Use fewer playouts for review mode
            c_puct=5.0,
            temperature=1.0,  # Use temperature=1 for review mode to show more variety
            training=False
        )

        # Convert visit counts to probabilities
        visit_counts = torch.zeros(game.BOARD_SIZE * game.BOARD_SIZE + 1, dtype=torch.float32, device=device)
        total_visits = 0
        for move, child in root.children.items():
            if move is None:
                visit_counts[-1] = root.N[move]  # Pass move
            else:
                idx = move[0] * game.BOARD_SIZE + move[1]
                visit_counts[idx] = root.N[move]
            total_visits += root.N[move]
        if total_visits > 0:
            visit_counts /= total_visits

        # Show policy heatmap
        fig_policy = plot_policy(game, visit_counts)
        display(fig_policy)

        # Show top-k suggested moves
        if total_visits > 0:
            print("\nTop suggested moves:")
            sorted_moves = sorted(root.children.items(), key=lambda x: root.N[x[0]], reverse=True)
            for i, (move, child) in enumerate(sorted_moves[:top_k]):
                if move is None:
                    print(f"{i+1}. Pass (visits: {root.N[move]})")
                else:
                    print(f"{i+1}. {move} (visits: {root.N[move]})")

        # Record evaluation and territory
        value_history.append(value)
        territory = game.estimate_territory()
        territory_history.append((territory['black_territory'], territory['white_territory']))

        # Show current evaluation
        print(f"Evaluation (value ∈ [-1,+1], +1=Black wins, -1=White wins): {value:.3f}")
    
        print(f"\nNetwork evaluation: {value:.3f} (Black winning: 1, White winning: -1)")
        print(f"Current territory - Black: {territory['black_territory']}, White: {territory['white_territory']}\n")

    # Game is over: show final board
    fig_final = plot_board(game)
    display(fig_final)

    # Show final score
    final_scores = game.score()
    print(f"Final score → Black: {final_scores['black_score']}, White: {final_scores['white_score']}")

    if return_moves:
        print(f"\nMove recording:")
        game.print_move_log()

    # Plot evaluation and score over move number
    moves = list(range(len(value_history)))
    black_scores = [t[0] for t in value_history]
    white_scores = [t[1] for t in value_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(moves, value_history, marker='o')
    ax1.set_title("Value (Win Estimate) over Moves")
    ax1.set_xlabel("Move Number")
    ax1.set_ylabel("Value (–1 to +1)")
    ax1.grid(True)

    ax2.plot(moves, black_scores, label="Black Score")
    ax2.plot(moves, white_scores, label="White Score")
    ax2.set_title("Score over Moves")
    ax2.set_xlabel("Move Number")
    ax2.set_ylabel("Score Count")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    display(fig)
    plt.close(fig)

    return game

