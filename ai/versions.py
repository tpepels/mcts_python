from math import sqrt, log

alpha = (
    beta
) = child_value = confidence_i = val_adj = ci_adjust = n_c = N = c = rand_fact = 0

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
