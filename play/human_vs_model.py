import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from boards.board_manager import GoGame
from models.policy_value_model import PolicyValueNet
from mcts.monte_carlo_tree_search_nodes import MCTSNode
from training.self_play_system import state_to_tensor, generate_influence_fields
# ----------------------------------------------------------------------
# Part 1: Plotting Helpers
# ----------------------------------------------------------------------

def plot_board(game: GoGame, ax=None):
    """
    Plot the current 19×19 Go board.
    Axes are labeled 0..18 on both x (columns) and y (rows), with (0,0) at bottom-left.
    Black stones are filled black circles, White stones are white circles with black edge.
    """
    board = game.board.numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    BOARD_SIZE = game.BOARD_SIZE
    BLACK = game.BLACK
    WHITE = game.WHITE

    # Set up ticks and labels
    ax.set_xticks(np.arange(BOARD_SIZE))
    ax.set_yticks(np.arange(BOARD_SIZE))
    ax.set_xticklabels([str(i) for i in range(BOARD_SIZE)])
    ax.set_yticklabels([str(i) for i in range(BOARD_SIZE)])
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')

    # Draw grid lines
    ax.grid(True, color='black', linewidth=0.5)

    # Draw stones: map (row=x, col=y) to (x-axis=y, y-axis=x)
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            val = board[x, y]
            if val == BLACK:
                circle = plt.Circle((y, x), 0.4, color='black')
                ax.add_patch(circle)
            elif val == WHITE:
                circle = plt.Circle((y, x), 0.4, edgecolor='black', facecolor='white')
                ax.add_patch(circle)

    plt.close(fig)
    return fig


def plot_policy(game: GoGame, policy_tensor: torch.Tensor, ax=None):
    """
    Overlay a heatmap of the policy distribution (length‐361+1 tensor) on the board grid.
    Axes labeled 0..18; origin at bottom-left. Uses float normalization.
    The last element of policy_tensor is the pass probability.
    """
    BOARD_SIZE = game.BOARD_SIZE
    # Convert to numpy float and reshape to [19,19] for board moves
    probs = policy_tensor.cpu().numpy().astype(np.float32)
    board_probs = probs[:-1].reshape((BOARD_SIZE, BOARD_SIZE))  # Exclude pass probability
    pass_prob = probs[-1]  # Get pass probability
    
    # Normalize board moves to [0,1] if not already
    total = board_probs.sum()
    if total > 0.0:
        board_probs = board_probs / total
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    # Show heatmap with origin="lower" so (0,0) is bottom-left
    im = ax.imshow(
        board_probs,
        cmap='hot',
        origin='lower',
        extent=[-0.5, BOARD_SIZE - 0.5, -0.5, BOARD_SIZE - 0.5],
        alpha=0.6
    )

    # Draw grid lines
    ax.set_xticks(np.arange(BOARD_SIZE))
    ax.set_yticks(np.arange(BOARD_SIZE))
    ax.set_xticklabels([str(i) for i in range(BOARD_SIZE)])
    ax.set_yticklabels([str(i) for i in range(BOARD_SIZE)])
    ax.grid(True, color='black', linewidth=0.5)
    ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
    ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
    ax.set_aspect('equal')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add pass probability text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f'Pass: {pass_prob:.3f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.close(fig)
    return fig


# ----------------------------------------------------------------------
# Part 2: Interactive Loop: Human vs. Network
# ----------------------------------------------------------------------

def get_user_move(game: GoGame):
    """
    Prompt the user to enter a move as "x y" (0-based). Returns (x,y) or None for pass.
    """
    while True:
        move_str = input("Enter your move as 'row col' (or 'pass'): ").strip().lower()
        if move_str == 'pass':
            return None
        if move_str == 'end game':
            return True
        parts = move_str.split()
        if len(parts) != 2:
            print("Invalid format. Please enter 'row col' or 'pass'.")
            continue
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("Invalid numbers. Try again.")
            continue
        if not (0 <= x < game.BOARD_SIZE and 0 <= y < game.BOARD_SIZE):
            print(f"Coordinates must be between 0 and {game.BOARD_SIZE-1}.")
            continue
        if not game.is_legal(x, y):
            print(f"Move ({x}, {y}) is illegal. Try again.")
            continue
        return (x, y)


def play_vs_net(policy_value_net: PolicyValueNet,
                device,
                num_playouts: int,
                c_puct: float,
                board_size: int,
                displays=False,
                return_moves = False,
                ):
    """
    Let a human play against the network. You choose Black or White, then
    loop until game over:
      - On human's turn: prompt for a move ("row column", "pass", or "end game").
      - On net's turn: run PUCT‐MCTS, show a policy heatmap, then apply argmax move.
    At each turn the board is re‐plotted so you can see the current position.
    """

    game = GoGame(board_size)
    BLACK = game.BLACK
    WHITE = game.WHITE
    BOARD_SIZE = game.BOARD_SIZE
    NUM_MOVES = BOARD_SIZE * BOARD_SIZE

    human_color = None
    while human_color not in ['b', 'w']:
        human_color = input("Choose your color ([B]lack or [W]hite): ").strip().lower()
    human_plays_black = (human_color == 'b')
    print(f"You are {'Black' if human_plays_black else 'White'}.\n")

    evals = []
    white_terrs = []
    black_terrs = []

    while not game.game_over:
        # Plot current board
        fig = plot_board(game)
        display(fig)

        # Determine whose turn
        is_human_turn = (game.current_player == BLACK and human_plays_black) or \
                        (game.current_player == WHITE and not human_plays_black)

        if is_human_turn:
            # Human's move
            user_move = get_user_move(game)
            if user_move is None:
                game.play_move()
                print("You passed.\n")
            elif user_move is True:
                game.game_over = True
                print(f"You ended the game.")
            else:
                game.play_move(user_move[0], user_move[1])
                print(f"You played at {user_move}.\n")
        else:
            # ---- Network's turn ----

            # a) Compute raw policy (no MCTS)
            state_tensor = state_to_tensor(game, device).unsqueeze(0)  # [1,2,19,19]
            state_tensor = torch.concat([state_tensor,
                                         generate_influence_fields(state_tensor, sigma = 1),
                                         generate_influence_fields(state_tensor, sigma = 3),
                                         generate_influence_fields(state_tensor, sigma = 6)], dim = 1)

            with torch.no_grad():
                raw_policy, eval = policy_value_net(state_tensor)  # [1,361], [1,1]
            raw_policy = raw_policy.squeeze(0)  # [361]
            evals.append(float(eval.item()))

            # b) Plot raw policy heatmap
            if displays:
                print("Network's turn—showing raw policy prior...\n")
                fig_raw = plot_policy(game, raw_policy)
                display(fig_raw)

            # c) Now run PUCT‐MCTS internally to pick a move
            root = MCTSNode(game.clone(), parent=None, move=None)
            for _ in range(num_playouts):
                node = root
                path = [node]

                # Selection
                while node.P is not None and not node.game.game_over:
                    mv, node = node.select_child(c_puct)
                    path.append(node)

                # Expansion & Evaluation
                if node.game.game_over:
                    terr = node.game.estimate_territory()
                    b_ter = terr['black_territory']
                    w_ter = terr['white_territory']
                    if b_ter > w_ter:
                        value_leaf = +1.0
                    elif w_ter > b_ter:
                        value_leaf = -1.0
                    else:
                        value_leaf = 0.0
                else:
                    value_leaf = node.expand_and_evaluate(policy_value_net, device)

                # Backpropagation
                for i in range(len(path) - 1):
                    parent = path[i]
                    child = path[i + 1]
                    mv_taken = child.move
                    parent.visits += 1
                    parent.N[mv_taken] += 1
                    parent.W[mv_taken] += value_leaf
                    parent.Q[mv_taken] = parent.W[mv_taken] / parent.N[mv_taken]
                    value_leaf = -value_leaf
                path[-1].visits += 1

            # Extract the visit‐count distribution π
            pi_tensor = torch.zeros(NUM_MOVES + 1, device=device)  # +1 for pass
            for mv, child in root.children.items():
                if mv is None:
                    pi_tensor[-1] = root.N[mv]  # Pass move is at the end
                else:
                    idx = mv[0] * BOARD_SIZE + mv[1]
                    pi_tensor[idx] = root.N[mv]
            if pi_tensor.sum() > 0:
                pi_tensor /= pi_tensor.sum()

            # Show policy heatmap
            if displays:
                print("Network is thinking with MCTS to pick its move...\n")
                fig2 = plot_policy(game, pi_tensor)
                display(fig2)

            # Pick argmax move
            top_idx = torch.argmax(pi_tensor).item()
            if pi_tensor[top_idx] == 0:
                net_move = None
            elif top_idx == NUM_MOVES:  # Pass move
                net_move = None
            else:
                net_move = (top_idx // BOARD_SIZE, top_idx % BOARD_SIZE)

            if net_move is None:
                game.play_move()  # pass
                terr = game.estimate_territory()

                black_terrs.append(terr['black_territory'])
                white_terrs.append(terr['white_territory'])

                print("Network passed.\n")
                if displays:
                    print(f"Scores: Black - {terr['black_territory']} | White - {terr['white_territory']}")
                    print(f"Network win-odds evaluation: {eval} (Black winning: 1, Wite winning: -1)")

            else:
                game.play_move(net_move[0], net_move[1])
                terr = game.estimate_territory()
                black_terrs.append(terr['black_territory'])
                white_terrs.append(terr['white_territory'])

                print(f"Network plays at {net_move}.\n")
                if displays:
                    print(f"Scores: Black - {terr['black_territory']} | White - {terr['white_territory']}")
                    print(f"Network win-odds evaluation: {eval} (Black winning: 1, Wite winning: -1)")

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
            print("Network wins!")
    elif w_ter > b_ter:
        print("White wins!")
        if not human_plays_black:
            print("You win!")
        else:
            print("Network wins!")
    else:
        print("It's a tie!")

    if return_moves:
        print(f"Move recording: \n")
        game.print_move_log()

    # After review ends, plot evaluation and territory over move number
    moves = list(range(len(evals)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(moves, evals, marker='o')
    ax1.set_title("Value (Win Estimate) over Moves")
    ax1.set_xlabel("Move Number")
    ax1.set_ylabel("Value (–1 to +1)")
    ax1.grid(True)

    ax2.plot(moves, black_terrs, label="Black Territory")
    ax2.plot(moves, white_terrs, label="White Territory")
    ax2.set_title("Estimated Territory over Moves")
    ax2.set_xlabel("Move Number")
    ax2.set_ylabel("Territory Count")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.close(fig)
    return fig

