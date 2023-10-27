DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C-value optimized for 9x9 5-in-a-row without improvements (0.8)
        "ai_params": {
            "c": 0.6,
            "imm_alpha": 0.2,
            "dyn_early_term_cutoff": 0.2,
            "pb_weight": 0.2,
        },
        "eval_params": {"m_power": 2, "m_centre": 3, "a": 200},
    },
    ("breakthrough", "mcts"): {
        # C-value optimized without improvements (0.8)
        "ai_params": {
            "c": 0.5,
            "imm_alpha": 0.6,
            "early_term_turns": 5,
            "early_term_cutoff": 0.1,
            "e_g_subset": 10,
            "epsilon": 0.1,
        },
        "eval_params": {"a": 300},
    },
}
