DEFAULT_SETTINGS = {
    # AMAZONS
    ("amazons", "mcts"): {
        "ai_params": {
            "c": 0.05,
            "imm_alpha": 0.6,
            "early_term_turns": 20,
            "early_term_cutoff": 0.1,
        },
        "eval_params": {},
    },
    # BREAKTHROUGH
    ("breakthrough", "mcts"): {
        "ai_params": {
            "c": 0.5,
            "imm_alpha": 0.2,
            "early_term_turns": 5,
            "early_term_cutoff": 0.15,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {},
    },
    # SHOGI
    ("minishogi", "mcts"): {
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.8,
            "early_term_turns": 15,
            "early_term_cutoff": 0.2,
            "epsilon": 0.03,
            "e_g_subset": 5,
        },
        "eval_params": {},
    },
    # GOMOKU
    ("ninarow", "mcts"): {
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.4,
            "dyn_early_term_cutoff": 0.4,
        },
        "eval_params": {},
    },
}
