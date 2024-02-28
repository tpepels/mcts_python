DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C_ab_imm: 0.1 (imm: 0.6)
        # C_imm: 0.1 (imm: 0.75)
        # C_vanilla: 0.3
        # C_ab: 0.3
        "ai_params": {
            "c": 0.1,
            "imm_alpha": 0.5,
            "dyn_early_term_cutoff": 0.4,
        },
        "eval_params": {},
    },
    ("breakthrough", "mcts"): {
        # c_ab_imm: 0.45 (imm 0.3)
        # c_imm: 0.4 (imm 0.3)
        # c_vanilla: 0.8
        # c_ab: 0.55
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.3,
            "early_term_turns": 10,
            # "pb_weight": 0.3,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {},
    },
    ("amazons", "mcts"): {
        # C_ab_imm: 0.1 (imm: 0.85)
        # C_imm: 0.1 (imm: 0.9)
        # C_vanilla: 0.5
        # C_ab: 0.4
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.8,
            "early_term_turns": 20,
            "early_term_cutoff": 0.3,
            # "e_g_subset": 10,
            # "epsilon": 0.1,
        },
        "eval_params": {},
    },
    ("kalah", "mcts"): {
        "ai_params": {"c": 0.4, "imm_alpha": 0.6},
        "eval_params": {"a": 300},
    },
}
