DEFAULT_SETTINGS = {
    ("minishogi", "mcts"): {
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.8,
            "early_term_turns": 20,
            "early_term_cutoff": 0.3,
            "epsilon": 0.03,
            "e_g_subset": 5,
        },
        "eval_params": {},
    },
    ("ninarow", "mcts"): {
        "ai_params": {
            "c": 0.2,
            "imm_alpha": 0.4,
            "dyn_early_term_cutoff": 0.4,
        },
        "eval_params": {},
    },
    ("breakthrough", "mcts"): {
        "ai_params": {
            "c": 0.5,
            "imm_alpha": 0.1,
            "early_term_turns": 10,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {},
    },
    ("amazons", "mcts"): {
        "ai_params": {
            "c": 0.05,
            "imm_alpha": 0.65,
            "early_term_turns": 20,
            "early_term_cutoff": 0.3,
        },
        "eval_params": {},
    },
}
