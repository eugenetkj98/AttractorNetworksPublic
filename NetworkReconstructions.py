import scipy as sp
import numpy as np
from tqdm import tqdm

# %% Tally up to calculate weighted flow adjacency matrix (Nearest match)
# EPSILON_FLOW is the maximum distance to define membership in a neighbourhood
# K_SCALE control the weighting based on distance
def states_to_M(states, states_history, EPSILON_FLOW, K_SCALE):

    M_flow = np.zeros((np.shape(states)[0],np.shape(states)[0]))
    M_convergence = []
    M_degrees = []
    M_edges = []

    P_flow_temp_1 = np.zeros((np.shape(states)[0],np.shape(states)[0]))
    P_flow_temp_2 = np.zeros((np.shape(states)[0],np.shape(states)[0]))

    for i in tqdm(range(np.shape(states_history)[0])):
        # Extract starting and ending position from raw data
        pos_1 = states_history[i,0,:]
        pos_2 = states_history[i,1,:]

        # Calculate distances between start/end positions and rest of possible states in attractor skeleton
        dist_1 = sp.spatial.distance_matrix([pos_1], states[:,0,:])[0]
        dist_2 = sp.spatial.distance_matrix([pos_2], states[:,0,:])[0]

        # Find closest point/distances in attractor skeleton to the start/end positions
        node_1 = np.argmin(dist_1)
        node_2 = np.argmin(dist_2)

        min_dist_1 = dist_1[node_1]
        min_dist_2 = dist_2[node_2]

        # Check if distance requirement is met
        if (min_dist_1 > EPSILON_FLOW) or (min_dist_2 > EPSILON_FLOW):
            continue
        else:
            # Calculate corresponding weight --> closer to skeleton = stronger weight (max deviation is 1)
            deviation = np.sqrt((min_dist_1/EPSILON_FLOW)**2+(min_dist_2/EPSILON_FLOW)**2)/np.sqrt(2)
            weight = np.exp(-K_SCALE*deviation)

            # Add weight to directed matrix
            M_flow[node_1, node_2] += weight
        
        if i == 0:
            nonzero_degrees = 0 # Count number of non_zero degrees
            for node_i in range(np.shape(states)[0]):
                if np.sum(M_flow[node_i,:]) > 0:
                    P_flow_temp_1[node_i,:] = M_flow[node_i,:]/np.sum(M_flow[node_i,:])
                    nonzero_degrees += 1
            # M_degrees.append(nonzero_degrees)
            # M_edges.append(np.sum(M_flow>0))
        else:
            if i%1000 == 0:
                nonzero_degrees = 0 # Count number of non_zero degrees
                for node_i in range(np.shape(states)[0]):
                    if np.sum(M_flow[node_i,:]) > 0:
                        P_flow_temp_2[node_i,:] = M_flow[node_i,:]/np.sum(M_flow[node_i,:])
                        nonzero_degrees += 1

                M_convergence.append(np.linalg.norm(P_flow_temp_1-P_flow_temp_2)/np.linalg.norm(P_flow_temp_1))
                M_degrees.append(nonzero_degrees)
                M_edges.append(np.sum(M_flow>0))
                P_flow_temp_1 = np.copy(P_flow_temp_2)

    # Remove self mappings
    np.fill_diagonal(M_flow, 0)

    return (M_flow, M_convergence, M_degrees, M_edges)
    
# %% Tally up to calculate weighted flow adjacency matrix (multiple weights)
# EPSILON_FLOW is the maximum distance to define membership in a neighbourhood
# K_SCALE control the weighting based on distance
def states_to_M_multiple(states, states_history, EPSILON_FLOW, K_SCALE, MAX_NODES = None):
    M_flow = np.zeros((np.shape(states)[0],np.shape(states)[0]))
    M_convergence = [0]
    M_degrees = []
    M_edges = []

    P_flow_temp_1 = np.zeros((np.shape(states)[0],np.shape(states)[0]))
    P_flow_temp_2 = np.zeros((np.shape(states)[0],np.shape(states)[0]))


    for i in tqdm(range(np.shape(states_history)[0])):
        # Extract starting and ending position from raw data
        pos_1 = states_history[i,0,:]
        pos_2 = states_history[i,1,:]

        # Calculate distances between start/end positions and rest of possible states in attractor skeleton
        dist_1 = sp.spatial.distance_matrix([pos_1], states[:,0,:])[0]
        dist_2 = sp.spatial.distance_matrix([pos_2], states[:,0,:])[0]

        if MAX_NODES is None:
            # Find all point/distances within EPSILON_FLOW in attractor skeleton to the start/end positions
            all_node_1 = np.where(dist_1<EPSILON_FLOW)[0]
            all_node_2 = np.where(dist_2<EPSILON_FLOW)[0]
        else:
            # Find at most MAX_NODES point/distances within EPSILON_FLOW in attractor skeleton to the start/end positions
            closest_idx_1 = np.argpartition(dist_1, MAX_NODES)[:MAX_NODES]
            closest_idx_2 = np.argpartition(dist_2, MAX_NODES)[:MAX_NODES]

            all_node_1 = closest_idx_1[np.where(dist_1[closest_idx_1]<EPSILON_FLOW)[0]]
            all_node_2 = closest_idx_2[np.where(dist_2[closest_idx_2]<EPSILON_FLOW)[0]]

        # Temporary matrix to hold edge update contributions
        M_flow_temp = np.zeros((len(all_node_1),len(all_node_2)))
        for I in range(len(all_node_1)):
            node_1 = all_node_1[I]
            for J in range(len(all_node_2)):
                node_2 = all_node_2[J]
                # Calculate corresponding weight --> closer to skeleton = stronger weight (max deviation is 1)
                deviation = np.sqrt((dist_1[node_1]/EPSILON_FLOW)**2+(dist_2[node_2]/EPSILON_FLOW)**2)/np.sqrt(2)
                weight = np.exp(-K_SCALE*deviation)

                # Add weight to directed matrix
                M_flow_temp[I, J] = weight
        
        M_flow_temps = M_flow_temp/np.sum(M_flow_temp)

        for I in range(len(all_node_1)):
            node_1 = all_node_1[I]
            for J in range(len(all_node_2)):
                node_2 = all_node_2[J]
                # Add weight to directed matrix
                M_flow[node_1, node_2] += M_flow_temp[I, J]
        
        if i == 1:
            nonzero_degrees = 0 # Count number of non_zero degrees
            for node_i in range(np.shape(states)[0]):
                if np.sum(M_flow[node_i,:]) > 0:
                    P_flow_temp_1[node_i,:] = M_flow[node_i,:]/np.sum(M_flow[node_i,:])
                    nonzero_degrees += 1
            M_degrees.append(nonzero_degrees)
            M_edges.append(np.sum(M_flow>0))
        else:
            if i%1000 == 0:
                nonzero_degrees = 0 # Count number of non_zero degrees
                for node_i in range(np.shape(states)[0]):
                    if np.sum(M_flow[node_i,:]) > 0:
                        P_flow_temp_2[node_i,:] = M_flow[node_i,:]/np.sum(M_flow[node_i,:])
                        nonzero_degrees += 1
                M_convergence.append(np.linalg.norm(P_flow_temp_1-P_flow_temp_2)/np.linalg.norm(P_flow_temp_1))
                M_degrees.append(nonzero_degrees)
                M_edges.append(np.sum(M_flow>0))
                P_flow_temp_1 = np.copy(P_flow_temp_2)

    
    # Remove self mappings
    np.fill_diagonal(M_flow, 0)

    return (M_flow, M_convergence, M_degrees, M_edges)

# %% Calculate level of surprise
# Helper Function
def calculate_surprise(test_state, trimmed_states, P_flow, EPSILON_FLOW, N, k_p = 2, ETA_CRITICAL = 2.5):
    # Constant of how much to scale the importance of "never seen before" events
    # k_p = 2

    # Starting and ending position of new data for testing
    pos_1 = test_state[0,:]
    pos_2 = test_state[1,:]

    # Calculate distances between start/end positions and rest of possible states in attractor skeleton
    dist_1 = sp.spatial.distance_matrix([pos_1], trimmed_states[:,0,:])[0]
    dist_2 = sp.spatial.distance_matrix([pos_2], trimmed_states[:,0,:])[0]

    best_i_node = np.argmin(dist_1)
    best_j_node = np.argmin(dist_2)

    # Check if closest point is within EPSILON_FLOW of attractor skeleton
    if dist_1[best_i_node] > EPSILON_FLOW:
        multiplier = 1+np.log(dist_1[best_i_node]/EPSILON_FLOW)
        eta = multiplier*ETA_CRITICAL # Assign baseline values of eta and importance p_ij
        p_ij = 1/(k_p*N)
    else:
        # Calculate node degrees
        degree_i = np.sum(P_flow[best_i_node,:]>0)

        # Calculate required entropies
        H_max_i = np.log(1/degree_i)

        P_i_idx = list(np.where(P_flow[best_i_node,:]>0)[0])
        P_i = P_flow[best_i_node, P_i_idx]
        H_i = np.sum(P_i*np.log(P_i))

        # Calculate meaningfulness eta
        if (H_max_i == 0) and (H_i == 0):
            eta = 1
        else:
            eta = H_max_i/H_i

        # Calculate probability of observation based on transition matrix
        # If observation has been made before when constructing the model:
        p_ij = P_flow[best_i_node, best_j_node]

        # If observation has never been encountered before, give it the probability
        # based on number of data points needed to construct model scaled by factor k_p
        if p_ij == 0:
            p_ij = 1/(k_p*N)

    # Calculate the associated "surprise" (entropy) corresponding to the obsevation
    H_transition = -np.log(p_ij)

    # Calculate final value for transition surprise scaled by node degree
    S_transition = eta*H_transition
    
    return (S_transition, p_ij, H_transition, eta)

