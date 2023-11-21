DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # imm value with alpha/beta: 0.15
        # imm value without alpha/beta: 0.15
        "ai_params": {
            "c": 0.6,
            "imm_alpha": 0.15,
            "dyn_early_term_cutoff": 0.2,
            # "pb_weight": 0.2,
        },
        "eval_params": {"m_power": 3, "m_centre": 3, "a": 500},
    },
    ("breakthrough", "mcts"): {
        # imm value with alpha/beta: 0.6
        # imm value without alpha/beta: 0.8
        "ai_params": {
            "c": 0.25,
            "imm_alpha": 0.8,
            "early_term_turns": 5,
            # "pb_weight": 0.3,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {"a": 60},
    },
    ("amazons", "mcts"): {
        # imm value with alpha/beta: 0.85
        # imm value without alpha/beta: 0.9
        "ai_params": {
            "c": 0.25,
            "imm_alpha": 0.85,
            "early_term_turns": 10,
            "early_term_cutoff": 0.05,
            # "e_g_subset": 10,
            # "epsilon": 0.1,
        },
        "eval_params": {"a": 50},
    },
    ("kalah", "mcts"): {
        "ai_params": {"c": 0.4, "imm_alpha": 0.6},
        "eval_params": {"a": 300},
    },
}
