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
        # C_ab_imm: 0.1 (imm: 0.6)
        # C_imm: 0.1 (imm: 0.75)
        # C_vanilla: 0.3
        # C_ab: 0.3
        "ai_params": {
            "c": 0.3,
            "imm_alpha": 0.3,
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
            "c": 0.5,
            "imm_alpha": 0.1,
            "early_term_turns": 10,
            # "pb_weight": 0.3,
            "early_term_cutoff": 0.05,
            "e_g_subset": 10,
            "epsilon": 0.05,
        },
        "eval_params": {},
    },
    ("amazons", "mcts"): {
        # These are tuned for 6x6 amazons (see below for 8x8 amazons results)
        "ai_params": {
            "c": 0.05,
            "imm_alpha": 0.9,
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

# +---------------------------------+----------------------+------------+-----------------------------------------+------------+------------+-----------+
# |               Exp.              |        Param1        | Wins (AI1) |                  Param2                 | Wins (AI2) | ± 95% C.I. | No. Games |
# +---------------------------------+----------------------+------------+-----------------------------------------+------------+------------+-----------+
# | amazons8_mcts_mcts_0315_2258_20 |    imm_alpha:0.8     |   38.42    |     ab_p1:2, ab_p2:2, imm_alpha:0.6     |   61.58    |   ±7.17    |    177    |
# | amazons8_mcts_mcts_0315_2258_21 |    imm_alpha:0.8     |   42.53    |     ab_p1:2, ab_p2:2, imm_alpha:0.7     |   57.47    |   ±7.35    |    174    |
# | amazons8_mcts_mcts_0315_2258_22 |                      |   44.77    |             ab_p1:2, ab_p2:2            |   55.23    |   ±7.43    |    172    |
# |  amazons8_mcts_mcts_0315_2258_1 | c:0.1, imm_alpha:0.8 |   45.09    |          c:0.05, imm_alpha:0.7          |   54.91    |   ±7.41    |    173    |
# | amazons8_mcts_mcts_0315_2258_19 | c:0.1, imm_alpha:0.8 |   45.40    | ab_p1:2, ab_p2:2, c:0.05, imm_alpha:0.9 |   54.60    |   ±7.40    |    174    |
# |  amazons8_mcts_mcts_0315_2258_2 |        c:0.1         |   46.15    |                  c:0.05                 |   53.85    |   ±7.52    |    169    |
