"""
Author: Eugene Tan (UWA Complex Systems Group)
Date: 7/6/2023
This code contains the scripts to run analyses for the Attractor Networks
change point detection method. This code runs the experiments presented in the paper
and must be adapted as appropriate for applying to other datasets.
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
User defined constants/parameters that control the construction of the attractor network
"""
ATTRACTOR_DATA_RATIO = 0.5 # Proportion of data to use for constructing the attractor network, remaining for test
EPSILON = 0.01 # Size/density scale of the attractor network nodes, as defined in paper
EPSILON_FLOW = 0.05 # Edge density of attractor network nodes, defined as delta in the paper
K_SCALE = 1 # Scaling parameter for weight of edges w.r.t distance, as defined int the paper
MIN_CLUSTERING = 0.5 # Maximum acceptable clustering coeffient to be considered a cluster
SUBSAMPLE = 1 # Subsampling data (optional)
SAMPLE_SIZE = int(3000/SUBSAMPLE)
MAX_DATA_LEN = 3800000
NETWORK_SIZE_LIMIT = 10000 # Approximate number of nodes to include in attractor skeleton before stopping.
SUPERSAMPLE = 1

# %%
"""
Import or generate data to analyse. This imrplementation generates phase Chua oscillators
"""
# Generate toy model data, normalise and extract time series component for embedding later
wash = 5000
T = 25000
dt = 0.02
func = CS.chua
dims = 3
data = CS.integrate(func, dt=dt, T=wash+T, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
data = data[::SUBSAMPLE]
lags = [13,38] # Selected lags used to contruct time delay vector

# %% 
"""
Normalise data and add noise if desired
"""
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

# %% Alternative EPSILON_FLOW CALCULATION based on distribution of points
inter_dist = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
smallest_dist = np.zeros(np.shape(inter_dist)[0])
for i in range(len(smallest_dist)):
    smallest_dist[i] = np.min(np.concatenate((inter_dist[i,:i],inter_dist[i,i+1:])))
EPSILON_FLOW = np.quantile(smallest_dist, 0.99)
print(EPSILON_FLOW)

# %%
"""
Construct dynamics component of attractor network
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

# # %% 
# """
# Plot nodes of attractor Network
# """
# M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
# M = np.multiply(M, M<EPSILON)
# G = nx.from_numpy_array(M)

# seed = 2468  # Seed random number generators for reproducibility
# plt.style.use('dark_background')
# # pos = nx.spring_layout(G, seed=seed, dim = 3)
# edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
# node_xyz = np.array([states[v,0,:] for v in sorted(G)])
# edge_xyz = np.array([(states[u,0,:], states[v,0,:]) for u, v in G.edges()])

# fig = plt.figure(figsize = (16,10))
# ax = fig.add_subplot(111, projection="3d")

# weights = weights/np.max(weights)

# # Plot the edges
# for i in range(len(edge_xyz)):
#     edge = edge_xyz[i]
#     ax.plot(*edge.T, color=plt.cm.winter(weights[i]))

# ax.w_xaxis.pane.fill = False
# ax.w_yaxis.pane.fill = False
# ax.w_zaxis.pane.fill = False

# %%
"""
Plot edges of attractor network
"""
G_flow = nx.from_numpy_array(M_flow)

seed = 2468  # Seed random number generators for reproducibility
plt.style.use('default')
# pos = nx.spring_layout(G, seed=seed, dim = 3)
edges,weights = zip(*nx.get_edge_attributes(G_flow,'weight').items())
node_xyz = np.array([states[v,0,:] for v in sorted(G_flow)])
edge_xyz = np.array([(states[u,0,:], states[v,0,:]) for u, v in G_flow.edges()])

fig = plt.figure(figsize = (12,10))
# ax = fig.add_subplot(111, projection="3d")
ax = fig.add_subplot(111)


# weights = weights/np.max(weights)
weights = weights/np.quantile(weights, 0.99)

# Plot the edges
for i in tqdm(range(len(edge_xyz))):
    edge = edge_xyz[i,:,[0,1]].T
    ax.plot(*edge.T, color=plt.cm.viridis(np.min([weights[i],1])), alpha = np.min([weights[i],1]), linewidth = 0.5)

# %%
"""
For testing against surrogate time series, generate freerun surrogate time series
with AAFT algorithm.
"""

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
ref_x = CS.integrate(func, dt=dt, T=wash+20000, RK = True, supersample = SUPERSAMPLE, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]

null_data = (ref_x - mu)/sigma # Normalise data
null_data = null_data[::SUBSAMPLE]
null_embed = CS.nonunif_embed2(null_data, lags)

# Null Hypothesis Data
test_embed_null = null_embed
test_series_null = np.empty((np.shape(test_embed_null)[0]-1, 2, np.shape(test_embed_null)[1]))
test_series_null[:,0,:] = test_embed_null[:-1,:]
test_series_null[:,1,:] = test_embed_null[1:,:]
# %% Calculate the expected values of surprise for the null hypothesis

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

# %% Calculate null hypothesis cutoffs for other comparison statistic methods
dx = 4
null_moving_average  = np.zeros(np.shape(test_series_null)[0])
null_moving_std = np.zeros(np.shape(test_series_null)[0])
null_moving_PE  = np.zeros(np.shape(test_series_null)[0])

for i in tqdm(range(bandwidth,np.shape(test_series_null)[0])):
    window = test_series_null[(i-bandwidth):i,0,0]
    null_moving_average[i] = np.mean(window)
    null_moving_std[i] = np.std(window)
    null_moving_PE[i] = OP.complexity_entropy(window, dx = dx, taux = round(lags[0]/SUBSAMPLE))[0]

# %% Tipping Point Detection Test Data (Single Channel)
"""Generate surrogate AAFT data to test method"""
# Tipping Point/Fluctuating Toggle Surrogate Data

switch_T = 1500 #Length of data between each toggle between normal and surrogate
n_switch = 7 # Number of toggles

for i in tqdm(range(n_switch)):
    if i%2 == 0: # Generate data for normal system
        if i == 0:
            generated_states = CS.integrate(func, dt=dt, T=wash+switch_T+1, RK = True, supersample = 1, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
        else:
            # Generate an initial state that links to the surrogate data
            append_len = 0
            while append_len < switch_T:
                append_candidate = CS.integrate(func, dt=dt, T=wash+10000, RK = True, supersample = 1, dims = dims, init = [0.5*np.random.rand(),0,0])[wash:,0]
                closest_match_idx = np.argmin(abs(append_candidate-generated_states[-1]))
                append_len = len(append_candidate)-closest_match_idx
            appended_states = append_candidate[closest_match_idx:closest_match_idx+switch_T]
            generated_states = np.append(generated_states, appended_states[1:], axis = 0)
    else: # Generate surrogate data
        surrogate_len = 0
        surrogate = np.copy(ref_x)
        while surrogate_len < switch_T:
            surrogate = SG.iter_alg2(ref_x)
            closest_match_idx = np.argmin(abs(surrogate-generated_states[-1]))
            surrogate_len = len(surrogate)-closest_match_idx
        appended_states = surrogate[closest_match_idx:closest_match_idx+switch_T]
        generated_states = np.append(generated_states, appended_states[1:], axis = 0)

generated_states = (generated_states - mu)/sigma # Normalise data
generated_states = generated_states[::SUBSAMPLE]

test_embed = CS.nonunif_embed2(generated_states, lags)
test_series = np.empty((np.shape(test_embed)[0]-1, 2, np.shape(test_embed)[1]))
test_series[:,0,:] = test_embed[:-1,:]
test_series[:,1,:] = test_embed[1:,:]

# Time values for plotting later
S_time = np.array(range(0,np.shape(test_series)[0]))

# # %% Tipping Point Detection Test Data (Single Channel)
# """Generate timeseries that toggles between PC and NPC Rossler"""

# switch_T = 4000
# n_switch = 7

# func2 = CS.rossler_NPC # Need to ensuree that func is previously set to be CS.rossler_PC

# for i in tqdm(range(n_switch)):
#     if i%2 == 0: # Generate data for normal system
#         if i == 0:
#             generated_states = CS.integrate(func, dt=dt, T=wash+switch_T, RK = True, supersample = 1, dims = dims)[wash:,:]
#         else:
#             # Generate an initial state that links to the surrogate data
#             appended_states = CS.integrate(func, dt=dt, T=switch_T+1, RK = True, supersample = 1, dims = dims, init = generated_states[-1,:])[1:,:]
#             generated_states = np.append(generated_states, appended_states, axis = 0)
#     else: # Generate surrogate data
#         appended_states = CS.integrate(func2, dt=dt, T=switch_T+1, RK = True, supersample = 1, dims = dims, init = generated_states[-1,:])[1:,:]
#         generated_states = np.append(generated_states, appended_states, axis = 0)

# generated_states = (generated_states[:,0] - mu)/sigma # Normalise data
# test_embed = CS.nonunif_embed2(generated_states, lags)[::SUBSAMPLE,:]

# test_series = np.empty((np.shape(test_embed)[0]-1, 2, np.shape(test_embed)[1]))
# test_series[:,0,:] = test_embed[:-1,:]
# test_series[:,1,:] = test_embed[1:,:]

# # Time values for plotting later
# S_time = np.array(range(0,np.shape(test_series)[0]))

# %% Calculate surprise to detect tipping point in data
"""
Calculate surprise levels of observed test data w.r.t to constructed attractor network
"""
S = []
eta = []
bandwidth = 100

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

# %% Calculate Moving Average, Standard Deviation, Permutation Entropy as comparison Diagonistic Tests
"""
Calculate comparison moving statistics on test data.
"""
dx = 4
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
    moving_PE[i] = OP.complexity_entropy(window, dx = dx, taux = round(lags[0]/SUBSAMPLE))[0]

# %% Calculate smoothed exponential w.r.t moving statistics
"""
Apply exponential smoothing to calculated statistics/metrics
"""
MA_cutoff = [np.quantile(null_moving_average, alpha/2),np.quantile(null_moving_average, 1-alpha/2)]
MSTD_cutoff = [np.quantile(null_moving_std, alpha/2),np.quantile(null_moving_std, 1-alpha/2)]
MPE_cutoff = [np.quantile(null_moving_PE, alpha/2),np.quantile(null_moving_PE, 1-alpha/2)]

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

# Moving Average
MA_exp_smooth, MA_smooth = exp_binary_statistic(moving_average, MA_cutoff, bandwidth = bandwidth)
MA_exp_smooth_null = exp_binary_statistic(null_moving_average, MA_cutoff, bandwidth = bandwidth)[0]
MA_exp_cutoff = 2*np.quantile(MA_exp_smooth_null, 0.95)
MA_binary = (MA_exp_smooth>MA_exp_cutoff).astype(int)

# Moving Standard Deviation
MSTD_exp_smooth, MSTD_smooth = exp_binary_statistic(moving_std, MSTD_cutoff, bandwidth = bandwidth)
MSTD_exp_smooth_null = exp_binary_statistic(null_moving_std, MSTD_cutoff, bandwidth = bandwidth)[0]
MSTD_exp_cutoff = 2*np.quantile(MSTD_exp_smooth_null, 0.95)
MSTD_binary = (MSTD_exp_smooth>MSTD_exp_cutoff).astype(int)

# Moving Permutation Entropy
MPE_exp_smooth, MPE_smooth = exp_binary_statistic(moving_PE, MPE_cutoff, bandwidth = bandwidth)
MPE_exp_smooth_null = exp_binary_statistic(null_moving_PE, MPE_cutoff, bandwidth = bandwidth)[0]
MPE_exp_cutoff = 2*np.quantile(MPE_exp_smooth_null, 0.95)
MPE_binary = (MPE_exp_smooth>MPE_exp_cutoff).astype(int)

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
fig, ax = plt.subplots(5, 1,figsize = (10,8), sharex = 'all')
fig.suptitle(f"Chua Detection (AAFS), Window = {bandwidth}", fontsize = 20)
# fig.suptitle(f"Rossler Detection (PC vs. NPC)", fontsize = 20)
ax0,ax1,ax2,ax3,ax4 = ax

lw = 1
lab_font_size = 18

ax0.plot(S_time[plot_start:plot_end]*dt, test_series[plot_start:plot_end,0,0], linewidth = lw)
ax0.set_ylabel("x(t)", fontsize = lab_font_size)

ax1.plot(S_time[plot_start:plot_end]*dt, moving_average[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax1.set_ylabel("MA", fontsize = lab_font_size)
ax1r = ax1.twinx()
ax1r.set_ylabel("E(t)", fontsize = lab_font_size)
ax1r.plot(S_time[plot_start:plot_end]*dt, MA_exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax1r.hlines(MA_exp_cutoff, S_time[plot_start]*dt, S_time[plot_end]*dt, color = "black", linestyle = '--')
ax1r.plot(S_time[plot_start:plot_end]*dt, MA_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)

ax2.plot(S_time[plot_start:plot_end]*dt, moving_std[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax2.set_ylabel("MSTD", fontsize = lab_font_size)
ax2r = ax2.twinx()
ax2r.set_ylabel("E(t)", fontsize = lab_font_size)
ax2r.plot(S_time[plot_start:plot_end]*dt, MSTD_exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax2r.hlines(MSTD_exp_cutoff, S_time[plot_start]*dt, S_time[plot_end]*dt, color = "black", linestyle = '--')
ax2r.plot(S_time[plot_start:plot_end]*dt, MSTD_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)

ax3.plot(S_time[plot_start:plot_end]*dt, moving_PE[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax3.set_ylabel("MPE", fontsize = lab_font_size)
ax3r = ax3.twinx()
ax3r.set_ylabel("E(t)", fontsize = lab_font_size)
ax3r.plot(S_time[plot_start:plot_end]*dt, MPE_exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax3r.hlines(MPE_exp_cutoff, S_time[plot_start]*dt, S_time[plot_end]*dt, color = "black", linestyle = '--')
ax3r.plot(S_time[plot_start:plot_end]*dt, MPE_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)

ax4.plot(S_time[plot_start:plot_end]*dt, S[plot_start:plot_end], linewidth = lw, alpha = 0.7)
ax4.set_ylabel("Surprise", fontsize = lab_font_size)
ax4r = ax4.twinx()
ax4r.set_ylabel("E(t)", fontsize = lab_font_size)
ax4r.plot(S_time[plot_start:plot_end]*dt, exp_smooth[plot_start:plot_end], linewidth = lw, color = 'black')
ax4r.hlines(exp_cutoff, S_time[plot_start]*dt, S_time[plot_end]*dt, color = "black", linestyle = '--')
ax4r.plot(S_time[plot_start:plot_end]*dt, S_binary[plot_start:plot_end], linewidth = 1.5, color = 'red', zorder = 0)
ax4r.legend(["E(t)","Cutoff", "Detection"], loc = 'upper right')


ax0.set_ylim(ax0.get_ylim()[0], ax0.get_ylim()[1])
ax1.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
ax1r.set_ylim(-0.05,1.05)
ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])
ax2r.set_ylim(-0.05,1.05)
ax3.set_ylim(ax3.get_ylim()[0], ax3.get_ylim()[1])
ax3r.set_ylim(-0.05,1.05)
ax4.set_ylim(ax4.get_ylim()[0], ax4.get_ylim()[1])
ax4r.set_ylim(-0.05,1.05)

for i in range(n_switch):
    if i%2!=0:
        xmin = dt*i*switch_T/SUBSAMPLE
        xmax = dt*(i+1)*switch_T/SUBSAMPLE
        col = 'orange'
        alpha = 0.1
        lw = 2
        ls = '--'
        ymin = 0
        ymax = 1
        ax0.axvspan(xmin, xmax, ymin=0, ymax=1, color = col, alpha = alpha)
        ax1.axvspan(xmin, xmax, ymin=0, ymax=1, color = col, alpha = alpha)
        ax2.axvspan(xmin, xmax, ymin=0, ymax=1, color = col, alpha = alpha)
        ax3.axvspan(xmin, xmax, ymin=0, ymax=1, color = col, alpha = alpha)
        ax4.axvspan(xmin, xmax, ymin=0, ymax=1, color = col, alpha = alpha)

        ax0.vlines([xmin,xmax], ymin = ax0.get_ylim()[0], ymax = ax0.get_ylim()[1], color = col, linewidth = lw, linestyle = ls)
        ax1.vlines([xmin,xmax], ymin = ax1.get_ylim()[0], ymax = ax1.get_ylim()[1], color = col, linewidth = lw, linestyle = ls)
        ax2.vlines([xmin,xmax], ymin = ax2.get_ylim()[0], ymax = ax2.get_ylim()[1], color = col, linewidth = lw, linestyle = ls)
        ax3.vlines([xmin,xmax], ymin = ax3.get_ylim()[0], ymax = ax3.get_ylim()[1], color = col, linewidth = lw, linestyle = ls)
        ax4.vlines([xmin,xmax], ymin = ax4.get_ylim()[0], ymax = ax4.get_ylim()[1], color = col, linewidth = lw, linestyle = ls)

fig.tight_layout()
# %%
