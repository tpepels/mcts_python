DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C_ab_imm: 0.4
        # C_imm: 0.4
        # C_vanilla: 0.5
        # C_ab: 0.4
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.35,
            "dyn_early_term_cutoff": 0.2,
        },
        "eval_params": {"m_power": 2, "m_centre": 3, "a": 400},
    },
    ("breakthrough", "mcts"): {
        # imm value with alpha/beta: 0.6
        # imm value without alpha/beta: 0.8
        "ai_params": {
            "c": 0.25,
            "imm_alpha": 0.8,
            "early_term_turns": 10,
            # "pb_weight": 0.3,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {"a": 120},
    },
    ("amazons", "mcts"): {
        # C_ab_imm - 0.5
        # C_imm - 0.25
        # C_vanilla - 0.6
        # C_ab - 0.5
        "ai_params": {
            "c": 0.25,
            "imm_alpha": 0.85,
            "early_term_turns": 10,
            "early_term_cutoff": 0.1,
            # "e_g_subset": 10,
            # "epsilon": 0.1,
        },
        "eval_params": {"a": 60},
    },
    ("kalah", "mcts"): {
        "ai_params": {"c": 0.4, "imm_alpha": 0.6},
        "eval_params": {"a": 300},
    },
}
