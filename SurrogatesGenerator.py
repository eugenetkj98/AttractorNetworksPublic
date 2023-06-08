# %%

import ChaoticSystems as CS
import ordpy as OP
import numpy as np

# %% Algorithm 0 (Shuffle Algorithm)
"""Surrogate series that preserves the mean and standard deviation"""
def alg0(x):
    surrogate = np.copy(x)
    np.random.shuffle(surrogate)

    return surrogate
# %% Algorithm 1 (Fourier Shuffled Algorithm)
"""Surrogate series that preserves the autocorrelation/Fourier power spectrum"""
def alg1(x):
    surrogate_freq = np.fft.rfft(x)
    phi = np.random.uniform(low = 0, high = 2*np.pi, size = len(surrogate_freq))
    new_freq = surrogate_freq*np.exp(phi*1j)
    surrogate = np.fft.irfft(new_freq)

    return surrogate

# %% Algorithm 2 (Amplitude Adjuster Fourier Transform)

"""Reorders time series y into the same order as reference time series x"""

def reorder(x,y):
    real_sorted_idx = np.argsort(x)
    y_sorted_idx = np.argsort(y)
    y_sorted_ascending = y[y_sorted_idx]
    y_sorted = np.zeros(len(y))
    for i in range(len(y_sorted)):
        y_sorted[real_sorted_idx[i]] = y_sorted_ascending[i]

    return y_sorted

# %%
def gamma_mismatch(x):
    gamma = np.zeros(len(x)-1)

    for i in range(len(gamma)-1):
        if i == 0:
            subseries = x
        else:
            subseries = x[:-i]
        gamma[i] = ((subseries[0]-subseries[-1])**2)/np.sum((subseries-np.mean(subseries))**2)
    
    return gamma
# %%

"""Surrogate series that approximately preserves both mean/std and autocorrelation/power spectrum"""

def alg2(x):
    # Ensure periodicities condition is met
    gamma = gamma_mismatch(x)
    idxs = len(x) - np.where(gamma<0.00004)[0]
    idx_end = idxs[np.where(idxs%2==0)[0][0]]
    x_alg = x[:idx_end]
    y = np.random.normal(size = len(x_alg))
    
    y = reorder(x_alg,y)
    y_hat = alg1(y)
    surrogate = reorder(y_hat, x_alg)

    return surrogate

# %%

def iter_alg2(x, iter = 10):
    # Ensure periodicities condition is met
    # max_range = np.max(x)-np.min(x)
    # idxs = np.where(abs(x-x[0])<max_range*0.001)[0]
    gamma = gamma_mismatch(x)
    idxs = len(x) - np.where(gamma<0.00004)[0]
    idx_end = idxs[np.where(idxs%2==0)[0][0]]
    x_alg = x[:idx_end]
    y = np.random.normal(size = len(x_alg))

    for i in range(iter):
        y = reorder(x_alg,y)
        y_hat = alg1(y)
        y = reorder(y_hat, x_alg)

    surrogate = np.copy(y)

    return surrogate

# %% 
"""Small-shuffle surrogates (Nakamura & Small, 2005)"""
def small_shuffle(ref_x, A = 1):
    # Extract matrix of indices
    i_t = np.array(range(len(ref_x)))

    # Perturb indices by a random amount with amplitude A
    i_perturbed = i_t + A*np.random.normal(size = len(ref_x))

    # Resort input data to create surrogate
    idx = np.argsort(i_perturbed)
    i_surrogate = i_t[idx]
    surrogate = np.copy(ref_x)[idx]
    return surrogate

# # %% Generate time series system switching signals (abrupt toggle with Alg2 surrogates)

# func_test = CS.lorenz
# wash = 2000
# switch_T = 1000
# n_switch = 10
# dt = 0.04
# dims = 3
# lags = [2,4]
# ref_x = CS.integrate(func_test, dt=dt, T=wash+20000, RK = True, supersample = 1, dims = dims)[wash:,0]



# # %%
# for i in tqdm(range(n_switch)):
#     if i%2 == 0: # Generate data for normal system
#         if i == 0:
#             generated_states = CS.integrate(func_test, dt=dt, T=wash+switch_T, RK = True, supersample = 1, dims = dims)[wash:,0]
#         else:
#             # Generate an initial state that links to the surrogate data
#             append_len = 0
#             while append_len < switch_T:
#                 append_candidate = CS.integrate(func_test, dt=dt, T=wash+10000, RK = True, supersample = 1, dims = dims, init = init_val)[wash:,0]
#                 closest_match_idx = np.argmin(abs(append_candidate-generated_states[-1]))
#                 append_len = len(append_candidate)-closest_match_idx
#             appended_states = append_candidate[closest_match_idx:closest_match_idx+switch_T]
#             generated_states = np.append(generated_states, appended_states, axis = 0)
#     else: # Generate surrogate data
#         surrogate_len = 0
#         surrogate = np.copy(ref_x)
#         while surrogate_len < switch_T:
#             surrogate = iter_alg2(ref_x)
#             closest_match_idx = np.argmin(abs(surrogate-generated_states[-1]))
#             surrogate_len = len(surrogate)-closest_match_idx
#         appended_states = surrogate[closest_match_idx:closest_match_idx+switch_T]
#         generated_states = np.append(generated_states, appended_states, axis = 0)

# # %%

# embedded_states = CS.nonunif_embed(generated_states, [2,4])
# switch_series = np.empty((np.shape(embedded_states)[0]-1, 2, np.shape(embedded_states)[1]))
# switch_series[:,0,:] = embedded_states[:-1,:]
# switch_series[:,1,:] = embedded_states[1:,:]

# # Normalise data
# for j in range(np.shape(switch_series)[2]):
#     switch_series[:,:,j] = (switch_series[:,:,j]-mu[j])/sigma[j]

# # %%
# bandwidth = 100
# moving_average  = np.zeros(np.shape(switch_series)[0])
# moving_std = np.zeros(np.shape(switch_series)[0])

# for i in tqdm(range(bandwidth,np.shape(switch_series)[0])):
#     moving_average[i] = np.mean(switch_series[(i-bandwidth):i])
#     moving_std[i] = np.std(switch_series[(i-bandwidth):i])
# %%
"""Method to calculate sliding window moving averages (looking backwards)
and trimming off initial washout"""
def moving_average(x, bandwidth = 10):
    moving_average = np.zeros(len(x)-bandwidth)
    for i in range(len(moving_average)):
        moving_average[i] = np.mean(x[i:i+bandwidth])
    return moving_average
# %% Experimental code to generate entropy preserved surrogates (Hirata, 2019)


def swap_elements(x, idx1, idx2):
    output = np.copy(x)
    element_1 = x[idx1]
    element_2 = x[idx2]
    output[idx1] = element_2
    output[idx2] = element_1

    return output

def match_permutation(pi_1, pi_2):
    comparison = pi_1!=pi_2

    return np.sum(np.sum(comparison, axis = 1)>0)

# # %%
# wash = 5000
# T = 20000
# dt = 0.01
# func = CS.lorenz
# dims = 3
# x = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = 1, dims = dims)[wash:,0]

# # %%
# fig = plt.figure(figsize = (16,8))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.plot(x[:5000])
# ax2.plot(iter_alg2(x)[:5000])



# # %%
# dx = 3
# taux = 2
# bandwidth = 10
# beta = 1/(2*iterations)
# iterations = 1000
# ma = moving_average(x, bandwidth = bandwidth)
# pi_s = OP.ordinal_sequence(x, dx = dx, taux = taux)
# moving_pi_s = OP.ordinal_sequence(ma, dx = dx, taux = taux)
# c_t = np.copy(x)
# probs = np.zeros(iterations)
# acceptance_count = 0
# for i in tqdm(range(iterations)):
#     # Create an attempted test surrogate by swapping two entries in the current time series
#     a_t = np.copy(c_t)
#     idx1, idx2 = np.random.choice(range(len(a_t)),size = 2, replace = False)
#     a_t = swap_elements(a_t, idx1, idx2)
#     a_t_ma = moving_average(a_t, bandwidth = bandwidth)
#     pi_a = OP.ordinal_sequence(a_t, dx = dx, taux = taux)
#     moving_pi_a = OP.ordinal_sequence(a_t_ma, dx = dx, taux = taux)
#     n_differences = match_permutation(pi_s,pi_a)+match_permutation(moving_pi_s,moving_pi_a)
#     acceptance_p = np.exp(-i*beta*n_differences)
#     probs[i] = acceptance_p
#     if np.random.uniform() < acceptance_p: # simulated annealing acceptance test
#         acceptance_count += 1
#         c_t = np.copy(a_t)

# %%
