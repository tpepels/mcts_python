import random
from ai.monte_carlo_tree_search import MCTS, UCTNode
from games.breakthrough import (
    BreakthroughGameState,
    evaluate as evaluate_breakthrough,
    visualize_breakthrough,
)
from games.amazons import AmazonsGameState, evaluate as evaluate_amazons, visualize_amazons


def run_experiment(game_type, player1, player2, num_games, **kwargs):
    if game_type == "breakthrough":
        game_state_cls = BreakthroughGameState
        visualization_fn = visualize_breakthrough
    elif game_type == "amazons":
        game_state_cls = AmazonsGameState
        visualization_fn = visualize_amazons
    else:
        raise ValueError(f"Unknown game type '{game_type}'")

    player1_wins = 0
    player2_wins = 0
    for game_idx in range(num_games):
        state = game_state_cls()
        while not state.is_terminal():
            if state.player == 1:
                action = player1(state, **kwargs)
            else:
                action = player2(state, **kwargs)
            state = state.apply_action(action)

        reward = state.get_reward()
        if reward == 1:
            player1_wins += 1
        elif reward == -1:
            player2_wins += 1

        print(f"Game {game_idx + 1} result: Player 1 wins - {player1_wins}, Player 2 wins - {player2_wins}")


def human_input(state, visualization_fn, *args, **kwargs):
    visualization_fn(state)
    action_str = input(
        "Enter your move (e.g., 'x1 y1 x2 y2 x3 y3' for Amazons or 'x1 y1 x2 y2' for Breakthrough): "
    )
    action = tuple(map(int, action_str.split()))
    return action


def mcts_play(state, num_simulations, evaluation_fn, *args, **kwargs):
    mcts = MCTS(UCTNode, state, num_simulations, evaluation_fn)
    return mcts.best_action()


def main():
    game_type = input("Enter game type ('breakthrough' or 'amazons'): ").strip()
    player1_type = input("Enter player 1 type ('human' or 'mcts'): ").strip()
    player2_type = input("Enter player 2 type ('human' or 'mcts'): ").strip()

    if game_type == "breakthrough":
        game_state_cls = BreakthroughGameState
        visualization_fn = visualize_breakthrough
        evaluation_fn = evaluate_breakthrough
    elif game_type == "amazons":
        game_state_cls = AmazonsGameState
        visualization_fn = visualize_amazons
        evaluation_fn = evaluate_amazons
    else:
        raise ValueError(f"Unknown game type '{game_type}'")

    if player1_type == "human":
        player1_fn = human_input
    elif player1_type == "mcts":
        player1_fn = mcts_play
    else:
        raise ValueError(f"Unknown player 1 type '{player1_type}'")

    if player2_type == "human":
        player2_fn = human_input
    elif player2_type == "mcts":
        player2_fn = mcts_play
    else:
        raise ValueError(f"Unknown player 2 type '{player2_type}'")

    if "mcts" in (player1_type, player2_type):
        num_simulations = int(input("Enter the number of MCTS simulations: ").strip())

    state = game_state_cls()
    while not state.is_terminal():
        if state.player == 1:
            action = (
                player1_fn(
                    state, visualization_fn, num_simulations=num_simulations, evaluation_fn=evaluation_fn
                )
                if player1_type == "mcts"
                else player1_fn(state, visualization_fn)
            )
        else:
            action = (
                player2_fn(
                    state, visualization_fn, num_simulations=num_simulations, evaluation_fn=evaluation_fn
                )
                if player2_type == "mcts"
                else player2_fn(state, visualization_fn)
            )
        state = state.apply_action(action)

    print("Game over!")
    if state.get_reward() == 1:
        print("Player 1 wins!")
    elif state.get_reward() == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
