DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C-value optimized for 9x9 5-in-a-row without improvements
        "ai_params": {"c": 0.8, "imm_alpha": 0.2, "dyn_early_term_cutoff": 0.2},
        "eval_params": {"m_power": 2, "m_centre": 3, "a": 200},
    },
    ("breakthrough", "mcts"): {
        # C-value optimized without improvements
        "ai_params": {"c": 0.8, "imm_alpha": 0.6},
        "eval_params": {"a": 300},
    },
}
