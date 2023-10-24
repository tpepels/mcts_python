DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C-value optimized for 9x9 5-in-a-row without improvements
        "ai_params": {"c": 0.8},
        "eval_params": {"m_power": 2, "m_centre": 3},
    },
    ("breakthrough", "mcts"): {
        # C-value optimized without improvements
        "ai_params": {"c": 0.8},
    },
}
