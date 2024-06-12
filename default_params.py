DEFAULT_SETTINGS = {
    # Optimized values for plain mcts with imm. Optimized on 22-3-2024
    # AMAZONS 8x8
    ("amazons", "mcts"): {
        "ai_params": {
            "c": 0.3,  # same without imm
            "imm_alpha": 0.2,
            "early_term_turns": 24,
            "early_term_cutoff": 0.03,
            "e_g_subset": 5,
            "epsilon": 0.005,
            "k_factor": 0.3,  # Same without imm
        },
        "eval_params": {"a": 60},
    },
    # BREAKTHROUGH
    ("breakthrough", "mcts"): {
        "ai_params": {
            "c": 0.4,  # 0.5 with imm
            "imm_alpha": 0.3,
            "early_term_turns": 4,
            "early_term_cutoff": 0.4,
            "e_g_subset": 15,
            "epsilon": 0.005,
            "k_factor": 0.4,  # Same for without imm
        },
        "eval_params": {},
    },
    # SHOGI
    ("minishogi", "mcts"): {
        "ai_params": {
            "c": 0.3,  # 0.4 with imm
            "imm_alpha": 0.4,
            "early_term_turns": 4,
            "early_term_cutoff": 0.1,
            "e_g_subset": 10,
            "epsilon": 0.01,
            "k_factor": 0.2,  # Same without imm
        },
        "eval_params": {},
    },
    # GOMOKU
    ("ninarow", "mcts"): {
        "ai_params": {
            "c": 0.6,  # 0.5 with imm
            "imm_alpha": 0.2,
            "early_term_turns": 16,
            "early_term_cutoff": 0.1,
            "mast": True,
            "epsilon": 0.3,
            "k_factor": 0.4,  # Same for without imm
        },
        "eval_params": {},
    },
}
# DEFAULT_SETTINGS = {
#     # Optimized values for plain mcts with imm. Optimized on 22-3-2024
#     # AMAZONS 8x8
#     ("amazons", "mcts"): {
#         "ai_params": {
#             "c": 0.3,  # same without imm
#             "imm_alpha": 0.2,
#             "early_term_turns": 24,
#             "early_term_cutoff": 0.03,
#             "e_g_subset": 5,
#             "epsilon": 0.005,
#             "k_factor": 0.3,  # Same without imm
#         },
#         "eval_params": {"a": 60},
#     },
#     # BREAKTHROUGH
#     ("breakthrough", "mcts"): {
#         "ai_params": {
#             "c": 0.5,  # 0.4 no imm
#             "imm_alpha": 0.3,
#             "early_term_turns": 4,
#             "early_term_cutoff": 0.4,
#             "e_g_subset": 15,
#             "epsilon": 0.005,
#             "k_factor": 0.4,  # Same for without imm
#         },
#         "eval_params": {},
#     },
#     # SHOGI
#     ("minishogi", "mcts"): {
#         "ai_params": {
#             "c": 0.4,  # 0.3 no imm
#             "imm_alpha": 0.4,
#             "early_term_turns": 6,
#             "early_term_cutoff": 0.4,
#             "e_g_subset": 10,
#             "epsilon": 0.01,
#             "k_factor": 0.2,  # Same without imm
#         },
#         "eval_params": {},
#     },
#     # GOMOKU
#     ("ninarow", "mcts"): {
#         "ai_params": {
#             "c": 0.5,  # 0.6 no imm
#             "imm_alpha": 0.2,
#             "early_term_turns": 16,
#             "early_term_cutoff": 0.1,
#             "mast": True,
#             "epsilon": 0.3,
#             "k_factor": 0.4,  # Same for without imm
#         },
#         "eval_params": {},
#     },
# }
