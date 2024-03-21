from math import sqrt, log

alpha = beta = child_value = confidence_i = val_adj = ci_adjust = n_c = N = c = rand_fact = 0

delta_alpha = child_value - alpha
delta_beta = beta - child_value
k = beta - alpha

# VERSIE 1: combo
if val_adj == 1:
    child_value = delta_alpha
if val_adj == 2:
    child_value = delta_beta
if val_adj == 3:
    child_value = 1 - (2 * (delta_alpha / k))
if val_adj == 4:
    child_value = 1 - abs(2 * (delta_alpha / k))
if val_adj == 5:
    child_value = 2 * (delta_alpha / k)
if val_adj == 6:
    child_value = max(delta_alpha, delta_beta)
if val_adj == 7:
    child_value = min(delta_alpha, delta_beta)
if val_adj == 8:
    child_value *= k

# * Ideen:
# c tuning voor sommige varianten met vermenigvuldiging
if ci_adjust == 1:
    confidence_i = max(c * confidence_i, k)
if ci_adjust == 2:
    confidence_i = min(c * confidence_i, k)
if ci_adjust == 3:
    confidence_i = max(c * confidence_i, max(abs(beta), abs(alpha)))
if ci_adjust == 4:
    confidence_i = min(c * confidence_i, max(abs(beta), abs(alpha)))
if ci_adjust == 5:
    confidence_i = k * c * confidence_i
if ci_adjust == 6:
    confidence_i = k * confidence_i
if ci_adjust == 7:
    confidence_i = c * sqrt((log(N) * k) / n_c) + rand_fact
if ci_adjust == 8:
    confidence_i = c * sqrt(log(N) / (n_c * k)) + rand_fact

# VERSIE 2: combo
if val_adj == 1:
    child_value = delta_alpha
if val_adj == 2:
    child_value *= k
if val_adj == 3:
    child_value += delta_alpha  # (Dit werkte eerst redeijk goed)

if val_adj == 4:  # ! Deze lijkt te werken
    child_value += delta_alpha * k

if val_adj == 5:
    child_value += k

if val_adj == 6:  # ! Deze lijkt te werken
    child_value = delta_alpha + k

if val_adj == 7:
    child_value = delta_alpha * delta_beta
if val_adj == 8:
    child_value = (delta_alpha * delta_beta) / k

if val_adj == 9:  # ! Deze lijkt te werken
    child_value = (child_value * k) + delta_alpha

# * Ideen:
# c tuning voor sommige varianten met vermenigvuldiging
if ci_adjust == 1:
    confidence_i = max(c * confidence_i, k)
if ci_adjust == 2:
    confidence_i = min(c * confidence_i, k)
if ci_adjust == 3:
    confidence_i = max(c * confidence_i, max(abs(beta), abs(alpha)))
if ci_adjust == 4:
    confidence_i = min(c * confidence_i, max(abs(beta), abs(alpha)))
if ci_adjust == 5:
    confidence_i = k * c * confidence_i
if ci_adjust == 6:
    confidence_i = k * confidence_i
if ci_adjust == 7:
    confidence_i = c * sqrt((log(N) * k) / n_c) + rand_fact
if ci_adjust == 8:
    confidence_i = c * sqrt(log(N) / (n_c * k)) + rand_fact

# Versie 3: combo
if val_adj == 1:
    child_value += delta_alpha * k
if val_adj == 2:
    child_value = delta_alpha + k
if val_adj == 3:
    child_value = (child_value * k) + delta_alpha
if val_adj == 4:
    child_value += delta_alpha / k
if val_adj == 5:
    child_value = delta_alpha + (1.0 / k)
if val_adj == 6:
    child_value = (child_value / k) + delta_alpha

if ci_adjust == 1:
    confidence_i = max(c * confidence_i, k)
if ci_adjust == 2:
    confidence_i = min(c * confidence_i, k)
if ci_adjust == 3:
    confidence_i = k * c * confidence_i
if ci_adjust == 4:
    confidence_i = (c / k) * confidence_i
if ci_adjust == 5:
    confidence_i = c * sqrt((log(N * k)) / n_c) + rand_fact
if ci_adjust == 6:
    confidence_i = c * sqrt((log(N) * k) / n_c) + rand_fact
if ci_adjust == 7:
    confidence_i = c * sqrt(log(N) / (n_c * k)) + rand_fact

# VERSIE 4

if val_adj == 1:
    child_value = child_value + (delta_alpha * k)
if val_adj == 2:
    child_value = (child_value * k) + delta_alpha
if val_adj == 3:
    child_value = (child_value + delta_alpha) * k
if val_adj == 4:
    child_value = delta_alpha + (2.0 / k)
if val_adj == 5:
    child_value = child_value + delta_alpha + (2.0 / k)
if val_adj == 6:
    child_value = child_value + delta_alpha
if val_adj == 7:
    child_value = child_value + (delta_alpha / k)
if val_adj == 8:
    child_value = (child_value + delta_alpha) / k
if val_adj == 9:
    child_value = (child_value / k) + delta_alpha

if ci_adjust == 1:
    confidence_i = c * sqrt((log(max(1, N * k))) / n_c) + rand_fact
if ci_adjust == 2:
    confidence_i = c * sqrt(log(N) / max(1, (n_c * k))) + rand_fact


# The parent's value have the information regarding the child's bound
val, bound = node.get_value_with_uct_interval(
    c=self.c,
    player=node.player,
    imm_alpha=self.imm_alpha,
    N=prev_node.n_visits,
)

if -val + bound <= beta[o_i] and -val - bound >= alpha[o_i] and val - bound > alpha[p_i] and val + bound < beta[p_i]:
    alpha[p_i] = val - bound
    beta[o_i] = -val + bound

elif alpha[p_i] != -INFINITY and beta[p_i] != INFINITY:
    prune = 1
    prunes += 1


# NA PRESENTATIE

if val_adj == 1:
    if child_value > alpha and child_value < beta:
        child_value += delta_alpha
if val_adj == 2:
    if child_value > alpha and child_value < beta:
        child_value += alpha + beta
if val_adj == 3:
    if child_value > alpha and child_value < beta:
        child_value += k
if val_adj == 4:
    if child_value > alpha or child_value < beta:
        child_value += beta

if val_adj == 5:
    if child_value > alpha and child_value < beta:
        child_value += delta_alpha
    else:
        # reset c
        c /= c_adjust
if val_adj == 6:
    if child_value > alpha and child_value < beta:
        child_value += alpha + beta
    else:
        # reset c
        c /= c_adjust
if val_adj == 7:
    if child_value > alpha and child_value < beta:
        child_value += k
    else:
        # reset c
        c /= c_adjust
if val_adj == 8:
    if child_value > alpha or child_value < beta:
        child_value += beta
    else:
        # reset c
        c /= c_adjust

# tries: cython.tuple = (
#     # 0 No data
#     (path_score, alpha, beta),
#     # 1 Worst Score: 27.87, Game: minishogi; Worst Score: 18.18, Game: breakthrough
#     (path_score, alpha, -beta),
#     # 2 Best Score: 56.77, Game: breakthrough; Best Score: 50.79, Game: minishogi
#     (path_score, -alpha, beta),
#     # 3 Worst Score: 49.21, Game: minishogi
#     (path_score, -alpha, -beta),
#     # 4 Worst Score: 32.08, Game: minishogi
#     (path_score, alpha_bounds, beta_bounds),
#     # 5 Worst Score: 49.74, Game: minishogi
#     (path_score, -alpha_bounds, beta_bounds),
#     # 6 Best Score: 59.38, Game: breakthrough; Best Score: 53.44, Game: minishogi
#     (path_score, -alpha_bounds, -beta_bounds),
#     # 7 Worst Score: 26.23, Game: breakthrough
#     (beta, alpha, -beta),
#     # 8 Worst Score: 23.53, Game: minishogi
#     (beta, -alpha, beta),
#     # 9 Best Score: 54.5, Game: minishogi; Best Score: 51.04, Game: breakthrough
#     (beta, -alpha, -beta),
#     # 10 Best Score: 53.65, Game: minishogi
#     (beta, alpha_bounds, -beta_bounds),
#     # 11 Best Score: 48.96, Game: breakthrough; Worst Score: 42.63, Game: minishogi
#     (beta, alpha_bounds, beta_bounds),
#     # 12 Worst Score: 30.3, Game: minishogi
#     (beta, -alpha_bounds, beta_bounds),
#     # 13 Best Score: 55.5, Game: minishogi
#     (beta, -alpha_bounds, -beta_bounds),
#     # 14 Worst Score: 18.75, Game: breakthrough
#     (alpha, alpha, -beta),
#     # 15 Best Score: 48.96, Game: breakthrough; Worst Score: 38.76, Game: minishogi
#     (alpha, -alpha, beta),
#     # 16 Best Score: 56.77, Game: breakthrough; Best Score: 52.13, Game: minishogi
#     (alpha, alpha_bounds, -beta_bounds),
#     # 17 Best Score: 54.74, Game: minishogi; Worst Score: 43.23, Game: breakthrough
#     (alpha, alpha_bounds, beta_bounds),
#     # 18 No data
#     (alpha, -alpha_bounds, beta_bounds),
#     # 19 No data
#     (alpha, -alpha_bounds, -beta_bounds),
# )
if imm_version == 0 or imm_version == 3:  # Plain old vanilla imm, just set the evaluation score
    # TODO If prog_bias is enabled, then we are computing the same thing twice
    child.im_value = new_state.evaluate(
        player=self.max_player,
        norm=True,
        params=eval_params,
    ) + uniform(-0.001, 0.001)
elif imm_version == 1 or imm_version == 13:  # n-ply-imm, set the evaluation score
    child.im_value = alpha_beta(
        state=new_state,
        eval_params=eval_params,
        max_player=self.max_player,
        depth=imm_ex_D,  # This is the depth that we will search
        alpha=-INFINITY,  # ? Can we set these to some meaningful values?
        beta=INFINITY,  # ? Can we set these to some more meaningful values?
    )

elif imm_version == 2 or imm_version == 23:  # q-imm, set the evaluation score to the quiescence score
    # TODO If prog_bias is enabled, then we are computing the same thing twice
    eval_value: cython.double = new_state.evaluate(
        player=self.max_player,
        norm=True,
        params=eval_params,
    ) + uniform(-0.001, 0.001)
    # Play out capture sequences
    if init_state.is_capture(action):
        child.im_value = quiescence(
            state=new_state,
            stand_pat=eval_value,
            eval_params=eval_params,
            max_player=self.max_player,
        )
    else:
        child.im_value = eval_value

elif imm_version == 12:  # n-ply-imm with quiescence
    # Play out capture sequences
    if init_state.is_capture(action):
        # TODO If prog_bias is enabled, then we are computing the same thing twice
        eval_value: cython.double = new_state.evaluate(
            player=self.max_player,
            norm=True,
            params=eval_params,
        ) + uniform(-0.001, 0.001)
        child.im_value = quiescence(
            state=new_state,
            stand_pat=eval_value,
            eval_params=eval_params,
            max_player=self.max_player,
        )
    else:
        child.im_value = alpha_beta(
            state=new_state,
            eval_params=eval_params,
            max_player=self.max_player,
            depth=imm_ex_D,  # This is the depth that we will search
            alpha=-INFINITY,  # ? Can we set these to some meaningful values?
            beta=INFINITY,  # ? Can we set these to some more meaningful values?
        )
