import argparse

# from tests import playout
from run_games import init_game
from playout_test import play_out

parser = argparse.ArgumentParser(description="Run test a playout for a game.")
parser.add_argument(
    "--game",
    choices=["amazons", "amazons8", "breakthrough", "ninarow59", "kalah66", "kalah86", "blokus", "minishogi"],
    default="ninarow",
    help="Choose the game (amazons, breakthrough, ninarow, kalah, blokus).",
)
args = parser.parse_args()

# Check if the game starts with "ninarow" or "kalah"
if args.game.startswith("ninarow"):
    game_name = "ninarow"
    row_length, board_size = int(args.game[-2]), int(args.game[-1])
    game_params = {"row_length": row_length, "board_size": board_size}
elif args.game.startswith("kalah"):
    game_name = "kalah"
    n_houses, init_seeds = int(args.game[-2]), int(args.game[-1])
    game_params = {"n_houses": n_houses, "init_seeds": init_seeds}
elif args.game.startswith("amazons") and args.game.endswith("8"):
    game_name = "amazons"
    board_size = int(args.game[-1])
    game_params = {
        "board_size": 8,
    }
else:
    game_name = args.game
    game_params = {}

game = init_game(game_name, game_params)
play_out(game)
