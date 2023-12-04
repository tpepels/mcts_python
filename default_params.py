DEFAULT_SETTINGS = {
    ("ninarow", "mcts"): {
        # C_ab_imm: 0.1 (imm: 0.6)
        # C_imm: 0.1 (imm: 0.75)
        # C_vanilla: 0.3
        # C_ab: 0.3
        "ai_params": {
            "c": 0.1,
            "imm_alpha": 0.75,
            "dyn_early_term_cutoff": 0.4,
        },
        "eval_params": {"m_power": 3, "m_centre": 4, "a": 700},
    },
    ("breakthrough", "mcts"): {
        # c_ab_imm: 0.45 (imm 0.3)
        # c_imm: 0.4 (imm 0.3)
        # c_vanilla: 0.8
        # c_ab: 0.55
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.3,
            "early_term_turns": 10,
            # "pb_weight": 0.3,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {
            "m_piece_diff": 0,
            "m_endgame": 0,
            "m_lorenz": 2,
            "m_safe": 2,
            "m_cap": 1,
            "m_mobility": 1,
            "m_decisive": 10,
            "a": 200,
        },
    },
    ("amazons", "mcts"): {
        # C_ab_imm: 0.1 (imm: 0.85)
        # C_imm: 0.1 (imm: 0.9)
        # C_vanilla: 0.5
        # C_ab: 0.4
        "ai_params": {
            "c": 0.1,
            "imm_alpha": 0.9,
            "early_term_turns": 20,
            "early_term_cutoff": 0.3,
            # "e_g_subset": 10,
            # "epsilon": 0.1,
        },
        "eval_params": {
            "n_moves_cutoff": 0,
            "m_imm": 1,
            "m_kill_s": 1,
            "m_min_mob": 2,
            "m_mob": 2,
            "a": 100,
        },
    },
    ("kalah", "mcts"): {
        "ai_params": {"c": 0.4, "imm_alpha": 0.6},
        "eval_params": {"a": 300},
    },
}
