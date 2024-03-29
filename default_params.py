DEFAULT_SETTINGS = {
    # Optimized values for plain mcts with imm. Optimized on 22-3-2024
    # AMAZONS 8x8
    ("amazons", "mcts"): {
        "ai_params": {
            "c": 0.3,
            "imm_alpha": 0.4,
            "early_term_turns": 10,
            "early_term_cutoff": 0.2,
        },
        "eval_params": {"a": 60},
    },
    # BREAKTHROUGH
    ("breakthrough", "mcts"): {
        "ai_params": {
            "c": 0.4,
            "imm_alpha": 0.4,
            "early_term_turns": 15,
            "early_term_cutoff": 0.3,
            # "e_g_subset": 10,
            # "epsilon": 0.05,
        },
        "eval_params": {},
    },
    # SHOGI
    ("minishogi", "mcts"): {
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.7,
            "early_term_turns": 5,
            "early_term_cutoff": 0.2,
            # "epsilon": 0.03,
            # "e_g_subset": 5,
        },
        "eval_params": {},
    },
    # GOMOKU
    ("ninarow", "mcts"): {
        "ai_params": {
            "c": 0.1,
            "imm_alpha": 0.4,
            "early_term_turns": 15,
            "early_term_cutoff": 0.2,
        },
        "eval_params": {},
    },
}
