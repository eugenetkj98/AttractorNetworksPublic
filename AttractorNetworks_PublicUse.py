"""
Author: Eugene Tan (UWA Complex Systems Group)
Last Updated: 28/5/2024
This code is an adapted version of AttractorNetworksRunAnalysis.py
that is modified for public use. Just fit in your own dataset and it will generate
the analysis outputs for change point detection as a final plot

Sections that contain options for users to change are noted with a (*)
"""

# %%
"""
Import relevant Python libraries
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
import ordpy as OP
from tqdm import tqdm
import pandas as pd
import json


# %%
"""
Import custom defined helper functions for running Attractor Network analyses
"""
import ChaoticSystems as CS
import AttractorNetworks as AN
import NetworkReconstructions as NR
import SurrogatesGenerator as SG

# %% 
"""
(*) User defined constants/parameters that control the construction of the attractor network
"""
ATTRACTOR_DATA_RATIO = 0.5 # Proportion of training data to use for constructing the attractor network, remaining for learning transitions
EPSILON = 0.01 # Size/density scale of the attractor network nodes, as defined in paper
# EPSILON_FLOW = 0.05 # Edge density of attractor network nodes, defined as delta in the paper
K_SCALE = 1 # Scaling parameter for weight of edges w.r.t distance, as defined int the paper
MIN_CLUSTERING = 0.5 # Maximum acceptable clustering coeffient to be considered a cluster
SUBSAMPLE = 1 # Subsampling data if wanting to downsample (optional)
SAMPLE_SIZE = int(3000/SUBSAMPLE)
MAX_DATA_LEN = 3800000
NETWORK_SIZE_LIMIT = 10000 # Approximate number of nodes to include in attractor skeleton before stopping.
SUPERSAMPLE = 1
bandwidth = 50 #width of sliding window when calculating moving statistics
cutoff_ratio = 2 # Multiplier w.r.t quantile of exponential smoothed curve to define onset of change point
quantile_alpha = 0.05 # Significance level for hypothesis testing portion to detect change points

# %%
"""
(*) Import or generate data to analyse. Data needs to be split into 3 components:
- Training healthy data: Used to construct the attractor network. Split between spatial and dynamics based on ATTRACTOR_DATA_RATIO
- Validation healthy data: Used to get the empirical cutoffs for what's considered 'healthy'
    (may be omitted, but then need to find a way to do change point detection based on just the surprise metric S(t))
- Test data: test data to run change point detection algorithm
"""


# Import data to be analysed
raw_data = raw_data
raw_data = np.reshape(raw_data, len(raw_data))
training_data = raw_data[:15000]
validation_data = raw_data[15000:]
test_data = raw_data[:]

# NOTE: This code block is an alternative example that does the same calculation using toy data. 
# wash = 5000
# T = 30000
# dt = 0.02
# func = CS.rossler_PC
# dims = 3
# raw_data = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
# training_data = raw_data[:15000]
# validation_data = raw_data[15000:]

# test_func = CS.rossler_NPC
# test_data_part_1 = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,:]
# test_data_part_2 = CS.integrate(test_func, dt=dt, T=T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = test_data_part_1[-1,:])[1:,:]
# test_data = np.concatenate((test_data_part_1, test_data_part_2))[:,0]

# Select lags required for delay embedding (e.g. SToPS, PECUZAL or Mutual Information Delays)
lags = [13,38] # Selected lags used to contruct time delay vector embedding
# %% 
"""
Normalise data and add noise if desired
"""

# Assign data
data = training_data
data = data[::SUBSAMPLE]

mu = np.min(data)
sigma = np.max(data)-np.min(data)
data = (data - mu)/sigma + 0.000001*np.random.randn(len(data))

embed = CS.nonunif_embed2(data, lags)

raw_states = np.zeros((np.shape(embed)[0]-1, 2, np.shape(embed)[1]))
raw_states[:,0,:] = embed[:-1,:]
raw_states[:,1,:] = embed[1:,:]

# Calculate amount of data needed for constructing spatial and dynamics network
n_attractor_data = int(ATTRACTOR_DATA_RATIO*np.shape(raw_states)[0])
n_flow_data = np.shape(raw_states)[0] - n_attractor_data

# %%
"""
Construct spatial component attractor network
"""

# Storage array for data
states = np.empty((0,2,np.shape(raw_states)[2]))
network_size = []
EPSILON_vals = []
original_len = np.shape(raw_states)[0]
COOLDOWN = 0
# Slow feed in samples of "New" data taken from recorded observations
while (np.shape(raw_states)[0] > n_flow_data):
    # Extract states from data
    sampled_idx = np.random.choice(range(np.shape(raw_states)[0]), SAMPLE_SIZE)
    temp = raw_states[sampled_idx, :,:]

    # Append to data collection
    states = np.append(states, temp, axis = 0)
    
    # Delete data from possible selected states
    raw_states = np.delete(raw_states, sampled_idx, axis = 0)

    # Trim attractor network
    states, M = AN.trim_attractor_network_clustering(states, epsilon = EPSILON, min_clustering = MIN_CLUSTERING)
    
    # Log size of attractor network
    network_size.append(np.shape(states)[0])
    EPSILON_vals.append(EPSILON)
    
    if (np.shape(raw_states)[0] < n_flow_data) or (np.shape(states)[0]>NETWORK_SIZE_LIMIT):
        if (np.shape(states)[0] > NETWORK_SIZE_LIMIT):
            print(f"Exceeded max network size of {NETWORK_SIZE_LIMIT} nodes.")
        print(f"Trimming Process Complete!")
        break
    else:
        print(f"Trimming Progress: {np.round(((original_len-np.shape(raw_states)[0])/n_attractor_data)*100, decimals = 2)}%")

# %% Calculate EPSILON_FLOW based on nearest neighbour distances in spatial attractor network
inter_dist = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
smallest_dist = np.zeros(np.shape(inter_dist)[0])
for i in range(len(smallest_dist)):
    smallest_dist[i] = np.min(np.concatenate((inter_dist[i,:i],inter_dist[i,i+1:])))
EPSILON_FLOW = np.quantile(smallest_dist, 0.99)
print(EPSILON_FLOW)

# %%
"""
(*) Construct dynamics component of attractor network
Can consider adjusting MAX_NODES
"""
states_history = np.copy(raw_states)
# %% Tally up to calculate weighted flow adjacency matrix (multiple weights)
# EPSILON_FLOW is the maximum distance to define membership in a neighbourhood
# K_SCALE controls the weighting based on distance
MAX_NODES = 6 # Maximum number of connected nodes from a given observed transition
M_flow, M_convergence, M_degrees, M_edges = NR.states_to_M_multiple(states, states_history, EPSILON_FLOW, K_SCALE, MAX_NODES = MAX_NODES)

# %% Calculate corresponding Transition Matrix for Surrogate Data Simulation
P_flow = np.copy(M_flow)
trimmed_states = np.copy(states)
trimmed_states, P_flow = AN.trim_flow_network_from_M(trimmed_states, P_flow)

# %%
"""
Get optimal ETA_CRITICAL corresponding to the level of surprise associated with making a previously unobserved transition
"""
# Use the average eta to scale cases where point does not lie within EPSILON_FLOW of attractor
eta_vals  = []
deg = np.sum(P_flow>0, axis = 1)
H_max = np.log(1/deg)
I_flow = P_flow*np.log(P_flow)
for i in tqdm(range(np.shape(I_flow)[0])):
    for j in range(np.shape(I_flow)[1]):
        if not (np.isnan(I_flow[i,j])):
            if (H_max[i] == 0) and (I_flow[i,j] == 0):
                eta_ij = 1
            else:
                eta_ij = H_max[i]/I_flow[i,j]
            eta_vals.append(eta_ij)

ETA_CRITICAL = np.mean(eta_vals)
print(ETA_CRITICAL)

# %% Generate Null hypothesis data to get cutoff parameters
validation_data = CS.integrate(func, dt=dt, T=wash+20000, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
ref_x = validation_data

null_data = (ref_x - mu)/sigma # Normalise data
null_data = null_data[::SUBSAMPLE]
null_embed = CS.nonunif_embed2(null_data, lags)

# Null Hypothesis Data
test_embed_null = null_embed
test_series_null = np.empty((np.shape(test_embed_null)[0]-1, 2, np.shape(test_embed_null)[1]))
test_series_null[:,0,:] = test_embed_null[:-1,:]
test_series_null[:,1,:] = test_embed_null[1:,:]

# %% 

"""
(*) Calculate the expected values of surprise for the null hypothesis
Can consider adjusting the S_quantile as the significance cutoff for flagging change points
"""

N = np.shape(data)[0]
S_null = []

for i in tqdm(range(0,np.shape(test_series_null)[0])):
    output = NR.calculate_surprise(test_series_null[i,:,:], trimmed_states, P_flow, EPSILON_FLOW,N, ETA_CRITICAL = ETA_CRITICAL)
    S_null.append(output[0])
# %% 
"""
(*) Below code same as above but also calculates null hypothesis values for comparison moving statistics (MA, MSTD, MPE)
"""
# Moving statistics don't need to separate training and validation. So just combine both to get null cutoffs
moving_stats_ref_x = np.concatenate((training_data, validation_data))

moving_stats_null_data = (moving_stats_ref_x - mu)/sigma # Normalise data
moving_stats_test_series_null = moving_stats_null_data[::SUBSAMPLE]

## Calculate above statistics for comparison MA, MSTD, MPE
MA_null = []
MSTD_null = []
dx = 4 # Dimension for calculating ordinal permutation entropy
MPE_null = []

for i in tqdm(range(bandwidth,np.shape(moving_stats_test_series_null)[0])):
    window = moving_stats_test_series_null[(i-bandwidth):i]
    MA_null.append(np.mean(window))
    MSTD_null.append(np.std(window))
    MPE_null.append(OP.complexity_entropy(window, dx = dx, taux = round(lags[-1]/dx))[0])

# %%
"""
Embed Test Data
"""
generated_states = (test_data - mu)/sigma # Normalise data
generated_states = generated_states[::SUBSAMPLE]

test_embed = CS.nonunif_embed2(generated_states, lags)
test_series = np.empty((np.shape(test_embed)[0]-1, 2, np.shape(test_embed)[1]))
test_series[:,0,:] = test_embed[:-1,:]
test_series[:,1,:] = test_embed[1:,:]

S_time = np.array(range(0,np.shape(test_series)[0]))

# %%
"""
(*) Calculate surprise levels of observed test data w.r.t to constructed attractor network
"""
S = []
eta = []
bandwidth = 100 #width of sliding window when calculating moving statistics

for i in tqdm(range(0,np.shape(test_series)[0])):
    output = NR.calculate_surprise(test_series[i,:,:], trimmed_states, P_flow, EPSILON_FLOW,N, ETA_CRITICAL = ETA_CRITICAL)
    S.append(output[0])
    eta.append(output[3])

S_q = np.zeros(len(S))
S_mean = np.zeros(len(S))
S_std = np.zeros(len(S)) 
for i in tqdm(range(bandwidth,np.shape(test_series)[0])):
    S_q[i] = np.quantile(S[(i-bandwidth):i], S_quantile)
    S_mean[i] = np.mean(S[(i-bandwidth):i])
    S_std[i] = np.std(S[(i-bandwidth):i])

S_time = np.array(range(0,np.shape(test_series)[0]))

# %%
"""
Calculate comparison moving statistics on test data.
"""
moving_average  = np.zeros(np.shape(test_series)[0])
moving_std = np.zeros(np.shape(test_series)[0])
moving_PE  = np.zeros(np.shape(test_series)[0])
moving_S = np.zeros(np.shape(S)[0])

for i in tqdm(range(bandwidth,np.shape(test_series)[0])):
    window = test_series[(i-bandwidth):i,0,0]
    S_window = S[(i-bandwidth):i]
    moving_average[i] = np.mean(window)
    moving_std[i] = np.std(window)
    moving_S[i] = np.mean(S_window)
    moving_PE[i] = OP.complexity_entropy(window, dx = dx, taux = round(lags[-1]/dx))[0]


# %% Helper Function for converting calculate metric/statistic into a normalised binary statistic
"""
This function converts a calculated, unscaled moving statistic into a 
new measure with exponential smoothing ranging from 0 to 1 (i.e. calculates E(t))
"""
def exp_binary_statistic(M, M_cutoff, bandwidth = 250):
    exp_alpha = 1/bandwidth
    M_binary_upper = (np.array(M)>M_cutoff[1]).astype(int)
    M_binary_lower = (np.array(M)<M_cutoff[0]).astype(int)
    M_binary = M_binary_upper + M_binary_lower
    exp_smooth = np.zeros(len(M_binary))
    M_smooth = np.zeros(len(M_binary))

    # Apply exponential smoothing
    for j in range(len(exp_smooth)):
        if j == 0:
            exp_smooth[j] = M_binary[j]
            M_smooth[j] = M[j]
        else:
            exp_smooth[j] = (1-exp_alpha)*exp_smooth[j-1] + exp_alpha*M_binary[j]
            M_smooth[j] = (1-exp_alpha)*M_smooth[j-1] + exp_alpha*M[j]
    
    return exp_smooth, M_smooth

"""
This function converts the smoothed E(t) into a binary 0 or 1 at every time step 
of whether an observation is normal (no change) or abnormal (change point)
with respect to some cutoff
"""
def exp_binary_surprise(S, S_cutoff, bandwidth = 250):
    exp_alpha = 1/bandwidth
    S_binary = (np.array(S)>S_cutoff).astype(int)
    exp_smooth = np.zeros(len(S_binary))
    S_smooth = np.zeros(len(S_binary))

    # Apply exponential smoothing
    for j in range(len(exp_smooth)):
        if j == 0:
            exp_smooth[j] = S_binary[j]
            S_smooth[j] = S[j]
        else:
            exp_smooth[j] = (1-exp_alpha)*exp_smooth[j-1] + exp_alpha*S_binary[j]
            S_smooth[j] = (1-exp_alpha)*S_smooth[j-1] + exp_alpha*S[j]
    
    return exp_smooth, S_smooth

def exp_binary_twoway(S, S_cutoff_lower, S_cutoff_upper, bandwidth = 250):
    exp_alpha = 1/bandwidth
    S_binary = np.maximum(((np.array(S)<S_cutoff_lower).astype(int)),((np.array(S)>S_cutoff_upper).astype(int)))
    exp_smooth = np.zeros(len(S_binary))
    S_smooth = np.zeros(len(S_binary))

    # Apply exponential smoothing
    for j in range(len(exp_smooth)):
        if j == 0:
            exp_smooth[j] = S_binary[j]
            S_smooth[j] = S[j]
        else:
            exp_smooth[j] = (1-exp_alpha)*exp_smooth[j-1] + exp_alpha*S_binary[j]
            S_smooth[j] = (1-exp_alpha)*S_smooth[j-1] + exp_alpha*S[j]
    
    return exp_smooth, S_smooth

# %%
"""
Calculate E(t) for various statistical measures (S, MA, MSTD, MPE). 
"""

# Attractor Network Surprise Metric
S_cutoff = np.quantile(S_null, 1-quantile_alpha)
exp_smooth, S_smooth = exp_binary_surprise(S, S_cutoff, bandwidth = bandwidth)
exp_smooth_null = exp_binary_surprise(S_null, S_cutoff, bandwidth = bandwidth)[0]
exp_cutoff = cutoff_ratio*np.quantile(exp_smooth_null, 1-quantile_alpha)
S_binary = (exp_smooth>exp_cutoff).astype(int)

# Moving Statistics Comparisons
MA_cutoff_lower = np.quantile(MA_null, quantile_alpha/2)
MA_cutoff_upper = np.quantile(MA_null, 1-quantile_alpha/2)
MA_exp_smooth, MA_smooth = exp_binary_twoway(moving_average[bandwidth+1:], MA_cutoff_lower, MA_cutoff_upper, bandwidth = bandwidth)
MA_exp_smooth_null = exp_binary_twoway(MA_null, MA_cutoff_lower, MA_cutoff_upper, bandwidth = bandwidth)[0]
MA_exp_cutoff = cutoff_ratio*np.quantile(MA_exp_smooth_null, 1-quantile_alpha)
MA_binary = (MA_exp_smooth>MA_exp_cutoff).astype(int)

MSTD_cutoff_lower = np.quantile(MSTD_null, quantile_alpha/2)
MSTD_cutoff_upper = np.quantile(MSTD_null, 1-quantile_alpha/2)
MSTD_exp_smooth, MSTD_smooth = exp_binary_twoway(moving_std[bandwidth+1:], MSTD_cutoff_lower, MSTD_cutoff_upper, bandwidth = bandwidth)
MSTD_exp_smooth_null = exp_binary_twoway(MSTD_null, MSTD_cutoff_lower, MSTD_cutoff_upper, bandwidth = bandwidth)[0]
MSTD_exp_cutoff = cutoff_ratio*np.quantile(MSTD_exp_smooth_null, 1-quantile_alpha)
MSTD_binary = (MSTD_exp_smooth>MSTD_exp_cutoff).astype(int)

MPE_cutoff_lower = np.quantile(MPE_null, quantile_alpha/2)
MPE_cutoff_upper = np.quantile(MPE_null, 1-quantile_alpha/2)
MPE_exp_smooth, MSTD_smooth = exp_binary_twoway(moving_PE[bandwidth+1:], MPE_cutoff_lower, MPE_cutoff_upper, bandwidth = bandwidth)
MPE_exp_smooth_null = exp_binary_twoway(MPE_null, MPE_cutoff_lower, MPE_cutoff_upper, bandwidth = bandwidth)[0]
MPE_exp_cutoff = cutoff_ratio*np.quantile(MPE_exp_smooth_null, 1-quantile_alpha)
MPE_binary = (MPE_exp_smooth>MPE_exp_cutoff).astype(int)

# %% Plot results

plot_start = bandwidth
plot_end = len(S_time)-1 #Subsampled

plt.style.use('default')
fig, ax = plt.subplots(5, 1,figsize = (10,8), sharex = 'all')
fig.suptitle(f"Test Detection, Window = {bandwidth}", fontsize = 20)
ax0,ax1,ax2,ax3,ax4 = ax

lw = 1
lab_font_size = 18

ax0.plot(S_time[plot_start:plot_end], test_series[plot_start:plot_end,0,0], linewidth = lw)
ax0.set_ylabel("x(t)", fontsize = lab_font_size)

ax1.plot(S_time[plot_start:plot_end], moving_average[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax1.set_ylabel("MA", fontsize = lab_font_size)
ax1r = ax1.twinx()
ax1r.set_ylabel("E_MA(t)", fontsize = lab_font_size)
ax1r.plot(S_time[plot_start:plot_end], MA_exp_smooth, linewidth = lw, color = 'black')
ax1r.hlines(MA_exp_cutoff, S_time[plot_start], S_time[plot_end], color = "black", linestyle = '--')
ax1r.plot(S_time[plot_start:plot_end], MA_binary, linewidth = 1.5, color = 'red', zorder = 0)

ax2.plot(S_time[plot_start:plot_end], moving_std[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax2.set_ylabel("MSTD", fontsize = lab_font_size)
ax2r = ax2.twinx()
ax2r.set_ylabel("E_MSTD(t)", fontsize = lab_font_size)
ax2r.plot(S_time[plot_start:plot_end], MSTD_exp_smooth, linewidth = lw, color = 'black')
ax2r.hlines(MSTD_exp_cutoff, S_time[plot_start], S_time[plot_end], color = "black", linestyle = '--')
ax2r.plot(S_time[plot_start:plot_end], MSTD_binary, linewidth = 1.5, color = 'red', zorder = 0)

ax3.plot(S_time[plot_start:plot_end], moving_PE[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax3.set_ylabel("MPE", fontsize = lab_font_size)
ax3r = ax3.twinx()
ax3r.set_ylabel("E_MPE(t)", fontsize = lab_font_size)
ax3r.plot(S_time[plot_start:plot_end], MPE_exp_smooth, linewidth = lw, color = 'black')
ax3r.hlines(MPE_exp_cutoff, S_time[plot_start], S_time[plot_end], color = "black", linestyle = '--')
ax3r.plot(S_time[plot_start:plot_end], MPE_binary, linewidth = 1.5, color = 'red', zorder = 0)

ax4.plot(S_time[plot_start:plot_end], S[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax4.set_ylabel("Surprise", fontsize = lab_font_size)
ax4r = ax4.twinx()
ax4r.set_ylabel("E(t)", fontsize = lab_font_size)
ax4r.plot(S_time[plot_start:plot_end], exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax4r.hlines(exp_cutoff, S_time[plot_start], S_time[plot_end], color = "black", linestyle = '--')
ax4r.plot(S_time[plot_start:plot_end], S_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)
ax4r.legend(["E(t)","Cutoff", "Detection"], loc = 'upper right')

ax0.set_ylim(ax0.get_ylim()[0], ax0.get_ylim()[1])
ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
ax1r.set_ylim(-0.05,1.05)

fig.tight_layout()
