"""
Author: Eugene Tan (UWA Complex Systems Group)
Date: 22/11/2023
This code is an adapted versiojn of AttractorNetworksRunAnalysis.py
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

# %%
"""
(*) Import or generate data to analyse. Data needs to be split into 3 components:
- Training healthy data: Used to construct the attractor network. Split between spatial and dynamics based on ATTRACTOR_DATA_RATIO
- Validation healthy data: Used to get the empirical cutoffs for what's considered 'healthy'
    (may be omitted, but then need to find a way to do change point detection based on just the surprise metric S(t))
- Test data: test data to run change point detection algorithm
"""
# Generate toy model data, normalise and extract time series component for embedding later (To be replaced by data import)
wash = 5000
T = 30000
dt = 0.02
func = CS.rossler_PC
dims = 3
raw_data = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
training_data = raw_data[:15000]
validation_data = raw_data[15000:]

test_func = CS.rossler_NPC
test_data_part_1 = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,:]
test_data_part_2 = CS.integrate(test_func, dt=dt, T=T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = test_data_part_1[-1,:])[1:,:]
test_data = np.concatenate((test_data_part_1, test_data_part_2))[:,0]

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

bandwidth = int(100/SUBSAMPLE)
S_quantile = 0.90
alpha = (1-S_quantile)/2

S_q_null = np.zeros(len(S_null))
S_mean_null = np.zeros(len(S_null))
S_std_null = np.zeros(len(S_null)) 

for i in tqdm(range(bandwidth,np.shape(test_series_null)[0])):
    S_q_null[i] = np.quantile(S_null[(i-bandwidth):i], S_quantile)
    S_mean_null[i] = np.mean(S_null[(i-bandwidth):i])
    S_std_null[i] = np.std(S_null[(i-bandwidth):i])

S_q_null_lower = np.quantile(S_q_null, alpha)
S_q_null_upper = np.quantile(S_q_null, 1-alpha)
S_q_null = np.median(S_q_null)

S_mean_null_lower = np.quantile(S_mean_null, alpha)
S_mean_null_upper = np.quantile(S_mean_null, 1-alpha)
S_mean_null = np.mean(S_mean_null)


S_std_null_lower = np.quantile(S_std_null, alpha)
S_std_null_upper = np.quantile(S_std_null, 1-alpha)
S_std_null = np.mean(S_std_null)

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
# dx = 4
# moving_average  = np.zeros(np.shape(test_series)[0])
# moving_std = np.zeros(np.shape(test_series)[0])
# moving_PE  = np.zeros(np.shape(test_series)[0])
moving_S = np.zeros(np.shape(S)[0])

for i in tqdm(range(bandwidth,np.shape(test_series)[0])):
    window = test_series[(i-bandwidth):i,0,0]
    S_window = S[(i-bandwidth):i]
    # moving_average[i] = np.mean(window)
    # moving_std[i] = np.std(window)
    moving_S[i] = np.mean(S_window)
    # moving_PE[i] = OP.complexity_entropy(window, dx = dx, taux = round(lags[0]/SUBSAMPLE))[0]


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

# %%
"""
Calculate E(t) for various statistical measures
"""

# Attractor Network Surprise Metric
S_cutoff = np.quantile(S_null, 0.95)
exp_smooth, S_smooth = exp_binary_surprise(S, S_cutoff, bandwidth = bandwidth)
exp_smooth_null = exp_binary_surprise(S_null, S_cutoff, bandwidth = bandwidth)[0]
exp_cutoff = 2*np.quantile(exp_smooth_null, 0.95)
S_binary = (exp_smooth>exp_cutoff).astype(int)

# %% Plot results

plot_start = bandwidth
plot_end = len(S_time)-1 #Subsampled

plt.style.use('default')
fig, ax = plt.subplots(2, 1,figsize = (10,8), sharex = 'all')
fig.suptitle(f"Test Detection, Window = {bandwidth}", fontsize = 20)
ax0,ax1 = ax

lw = 1
lab_font_size = 18

ax0.plot(S_time[plot_start:plot_end]*dt, test_series[plot_start:plot_end,0,0], linewidth = lw)
ax0.set_ylabel("x(t)", fontsize = lab_font_size)

ax1.plot(S_time[plot_start:plot_end]*dt, S[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax1.set_ylabel("Surprise", fontsize = lab_font_size)
ax1r = ax1.twinx()
ax1r.set_ylabel("E(t)", fontsize = lab_font_size)
ax1r.plot(S_time[plot_start:plot_end]*dt, exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax1r.hlines(exp_cutoff, S_time[plot_start]*dt, S_time[plot_end]*dt, color = "black", linestyle = '--')
ax1r.plot(S_time[plot_start:plot_end]*dt, S_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)
ax1r.legend(["E(t)","Cutoff", "Detection"], loc = 'upper right')

ax0.set_ylim(ax0.get_ylim()[0], ax0.get_ylim()[1])
ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
ax1r.set_ylim(-0.05,1.05)

fig.tight_layout()
