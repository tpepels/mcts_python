DEFAULT_SETTINGS = {
    # Optimized values for plain mcts with imm. Optimized on 22-3-2024
    # AMAZONS 8x8
    ("amazons", "mcts"): {
        "ai_params": {
            "c": 0.3,
            "imm_alpha": 0.2,
            "early_term_turns": 24,
            "early_term_cutoff": 0.03,
            "e_g_subset": 5,
            "epsilon": 0.005,
        },
        "eval_params": {"a": 60},
    },
    # BREAKTHROUGH
    ("breakthrough", "mcts"): {
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.3,
            "early_term_turns": 4,
            "early_term_cutoff": 0.4,
            "e_g_subset": 15,
            "epsilon": 0.005,
        },
        "eval_params": {},
    },
    # SHOGI
    ("minishogi", "mcts"): {
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.3,
            "early_term_turns": 6,
            "early_term_cutoff": 0.4,
            "e_g_subset": 10,
            "epsilon": 0.01,
        },
        "eval_params": {},
    },
    # GOMOKU
    ("ninarow", "mcts"): {
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.3,
            "early_term_turns": 16,
            "early_term_cutoff": 0.1,
            "mast": True,
            "epsilon": 0.3,
        },
        "eval_params": {},
    },
}
