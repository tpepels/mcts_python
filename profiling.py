from run_games import run_game, AIParams
import cProfile

p1_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_amazons_lieberum",
    max_player=1,
    ai_params={"max_time": 10, "debug": True},
)
p2_params = AIParams(
    ai_key="alphabeta",
    eval_key="evaluate_amazons_lieberum",
    max_player=2,
    ai_params={"max_time": 10, "debug": True},
)

cProfile.run('run_game(game_key="amazons", game_params={}, p1_params=p1_params, p2_params=p2_params)')
