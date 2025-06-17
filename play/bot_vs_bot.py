import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.run_batched_mcts import run_batched_mcts
from training.self_play_system import state_to_tensor, generate_influence_fields
from play.human_vs_model import plot_board, plot_policy


def bot_vs_bot(
        black_net: PolicyValueNet,
        white_net: PolicyValueNet,
        device: torch.device,
        num_playouts: int,
        c_puct: float,
        board_size: int,
        displays: bool = False,
        return_moves: bool = False,
        temperature: float = 0.0,  # 0.0 for deterministic play, 1.0 for exploration
) -> dict:
    """
    Let two PolicyValueNet models play against each other.
    
    Args:
        black_net: PolicyValueNet model playing as Black
        white_net: PolicyValueNet model playing as White
        device: Torch device to use
        num_playouts: Number of MCTS playouts per move
        c_puct: Exploration constant for MCTS
        board_size: Size of the Go board
        displays: Whether to show policy heatmaps and board state
        return_moves: Whether to return the move history
        temperature: Controls move selection randomness (0.0=deterministic, 1.0=exploration)
        
    Returns:
        Dictionary containing:
        - winner: "black", "white", or "tie"
        - final_score: dict with black_score and white_score
        - move_history: list of moves if return_moves=True
        - evaluation_history: list of evaluation values
        - black_scores: list of black scores over time
        - white_scores: list of white scores over time
    """
    game = GoGame(board_size)
    BLACK = game.BLACK
    WHITE = game.WHITE
    BOARD_SIZE = game.BOARD_SIZE
    NUM_MOVES = BOARD_SIZE * BOARD_SIZE

    evals = []
    black_scores = []
    white_scores = []
    move_history = []

    while not game.game_over:
        # Plot current board if requested
        if displays:
            fig = plot_board(game)
            display(fig)

        # Determine which net to use based on current player
        current_net = black_net if game.current_player == BLACK else white_net

        # a) Compute raw policy (no MCTS)
        state_tensor = state_to_tensor(game, device).unsqueeze(0)  # [1,6,19,19]
        state_tensor = torch.concat([state_tensor,
                                   generate_influence_fields(state_tensor, sigma=1),
                                   generate_influence_fields(state_tensor, sigma=3),
                                   generate_influence_fields(state_tensor, sigma=6)
                                  ], dim=1)

        with torch.no_grad():
            raw_policy, eval = current_net(state_tensor)  # [1,361], [1,1]
        raw_policy = raw_policy.squeeze(0)  # [361]
        evals.append(float(eval.item()))

        # b) Plot raw policy heatmap if requested
        if displays:
            print(f"{'Black' if game.current_player == BLACK else 'White'}'s turn—showing raw policy prior...\n")
            fig_raw = plot_policy(game, raw_policy)
            display(fig_raw)

        # c) Run batched MCTS to pick a move
        best_move, root = run_batched_mcts(
            game.clone(),
            current_net,
            device,
            num_playouts,
            c_puct,
            temperature=temperature,
            training=False
        )

        # d) Play the selected move
        if best_move is None:
            game.play_move()  # pass
            move_history.append(("pass", game.current_player))
            if displays:
                print(f"{'Black' if game.current_player == BLACK else 'White'} passed.\n")
        else:
            game.play_move(best_move[0], best_move[1])
            move_history.append((best_move, game.current_player))
            if displays:
                print(f"{'Black' if game.current_player == BLACK else 'White'} played at {best_move}.\n")

        # e) Show MCTS policy if requested
        if displays:
            # Convert visit counts to probabilities
            visit_counts = torch.zeros(NUM_MOVES + 1, dtype=torch.float32, device=device)
            total_visits = 0
            for move, child in root.children.items():
                if move is None:
                    visit_counts[-1] = root.N[move]  # Pass move
                else:
                    idx = move[0] * BOARD_SIZE + move[1]
                    visit_counts[idx] = root.N[move]
                total_visits += root.N[move]
            if total_visits > 0:
                visit_counts /= total_visits

            print(f"{'Black' if game.current_player == BLACK else 'White'}'s turn—showing MCTS policy...\n")
            fig_mcts = plot_policy(game, visit_counts)
            display(fig_mcts)

        # Track scores after each move
        score = game.score()
        black_scores.append(score['black_score'])
        white_scores.append(score['white_score'])

        if displays:
            print(f"Evaluation (value ∈ [-1,+1], +1=Black wins, -1=White wins): {float(eval.item()):.3f}")
            print(f"Current territory - Black: {score['black_score']}, White: {score['white_score']}\n")

    # Game is over: show final board and result
    if displays:
        fig_final = plot_board(game)
        display(fig_final)

    final_score = game.score()
    b_score = final_score['black_score']
    w_score = final_score['white_score']
    
    # Determine winner
    if b_score > w_score:
        winner = "black"
    elif w_score > b_score:
        winner = "white"
    else:
        winner = "tie"

    if displays:
        print(f"Final score → Black: {b_score}, White: {w_score}")
        print(f"{winner.capitalize()} wins!" if winner != "tie" else "It's a tie!")

    if return_moves:
        print(f"Move recording: \n")
        game.print_move_log()

    # Plot evaluation and territory over move number if requested
    if displays:
        moves = list(range(len(evals)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot evaluation values
        ax1.plot(moves, evals, marker='o')
        ax1.set_title("Value (Win Estimate) over Moves")
        ax1.set_xlabel("Move Number")
        ax1.set_ylabel("Value (–1 to +1)")
        ax1.grid(True)

        # Plot scores from the stored lists
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

    # Return match results
    results = {
        "winner": winner,
        "final_score": final_score,
        "evaluation_history": evals,
        "black_scores": black_scores,
        "white_scores": white_scores
    }
    
    if return_moves:
        results["move_history"] = move_history

    return results

