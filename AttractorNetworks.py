# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sp
import ChaoticSystems as CS
from tqdm import tqdm

'''Helper function to quickly calculate the presence of unwanted simplices from new
node states when trying to simplify'''
def append_distance_matrix(M, new_state, current_states, epsilon):

    # Quickly calculate pairwise distances w.r.t new state
    M_new = np.pad(M, ((0,1),(0,1)))

    new_pos = new_state[0,0,:]
    reps = np.tile(new_pos,[np.shape(current_states)[0],1])
    dists = np.sqrt(np.sum((current_states[:,0,:]-reps)**2, axis = 1))
    dists = np.multiply(dists, dists<epsilon)

    M_new[0:-1,-1] = dists
    M_new[:-1,0:-1] = dists

    return M_new

"""
Function to enumerate cliques from weighted adjacency distance matrix M
"""
def count_cliques_from_M(M, printout = False):
    G = nx.from_numpy_array(M)
    it = nx.enumerate_all_cliques(G)
    clique_count = np.zeros(0, dtype = int)
    clique_lists = []

    # for i in range(k_max):
    #     clique_lists.append([])

    k_max = 0
    k = 0
    while k >= 0:
        try:
            clique = next(it)
            k = len(clique)
            if k>k_max: # New maximal clique found
                if printout == True:
                    print(f"Found new maximal clique of size {k}")
                clique_lists.append([]) # Expand number of entries
                clique_count = np.append(clique_count, 0)
                k_max = k

            if k <= 2: # Ignore individual nodes and edges
                continue
            else:
                clique_count[k-1] += 1
                clique_lists[k-1].append(clique)
        except:
            break
    return clique_count


"""
Function to append new data to collection of points (states)
Lags toggles embedded vs non-embedded mode.
"""
def feed_new_data(func, dt, epsilon = 0.8, N_NEW_SIMULATIONS = 4, wash = 2000, T = 500, 
                    isrossler = False, mu = 0, sigma = 1, dims = 3, lags = 0):
    print("Generating New Data...")

    # Different operation if using delay embedding or not
    if lags == 0:
        new_states = np.empty((0,2,dims))

    for i in tqdm(range(N_NEW_SIMULATIONS)):
        # Generate states and temporarily store in array
        generated_states = CS.integrate(func, dt=0.02, T=wash+T, RK = True, supersample = 1, dims = dims)[wash:,:]
        
        # Only include for Rossler data
        if isrossler:
            generated_states[:,2] = np.log(generated_states[:,2])
        
        # Use full state space
        if lags == 0:
            temp = np.empty((np.shape(generated_states)[0]-1, 2, np.shape(generated_states)[1]))
            temp[:,0,:] = generated_states[:-1,:]
            temp[:,1,:] = generated_states[1:,:]
        
        # Use delay embedding
        else:
            # Take only the first component
            generated_states = CS.integrate(func, dt=0.02, T=wash+T, RK = True, supersample = 1, dims = dims)[wash:,0]
            generated_states = CS.nonunif_embed(generated_states, lags)

            temp = np.empty((np.shape(generated_states)[0]-1, 2, np.shape(generated_states)[1]))
            temp[:,0,:] = generated_states[:-1,:]
            temp[:,1,:] = generated_states[1:,:]

        # Normalise
        for j in range(np.shape(temp)[2]):
            temp[:,:,j] = (temp[:,:,j]-mu[j])/sigma[j]

        # Append to data collection
        new_states = np.append(new_states, temp, axis = 0)

    return new_states

"""
Function that takes in a list of states and trims down the attractor network by replacing
simplices with centre of masses (averages)
"""
def trim_attractor_network(states, epsilon = 0.8):
    print("Trimming Attractor Network...")
    M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
    M = np.multiply(M, M<epsilon)
    G = nx.from_numpy_array(M)

    clique_counts = count_cliques_from_M(M)
    num_cliques = np.sum(clique_counts)

    while num_cliques>0:
        # Storage for new states and list of nodes to delete
        new_states = np.zeros((0,2,np.shape(states)[2]))
        redundant_nodes = []

        # Find highest dimension clique to replace:
        k_target = np.amax(np.where(clique_counts>0)[0]) + 1 # +1 to account for python 0 indexing

        # Start trimming process
        it = nx.enumerate_all_cliques(G) # Create generator for cliques

        k = 1 # Initialisation k
        k_max = k_target+1 # termination clique size to prevent unecessary computation
        M_temp = np.zeros((0,0)) # Temporary distance matrix used to calculate positions of new nodes
        counter = 0 # Counter to track progress

        while k < k_max and k > 0:
            try:
                clique = next(it)
                k = len(clique)
                if k == k_target: # Only take cases where the clique is of the desired size.
                    counter = counter + 1
                    if counter%100 == 0:
                        progress = counter/clique_counts[k_target-1]
                        print(f'Simplification Progress: {np.round(progress*100, decimals = 3)}%')
                    # Simplifying Simplices with averages
                    # Initially accept multiple states to populate new states list
                    if np.shape(new_states)[0] < 2:
                        simplice_positions = np.array([np.mean(states[clique,:,:], axis = 0)])
                        new_states = np.append(new_states, simplice_positions, axis = 0)
                        redundant_nodes = redundant_nodes + clique
                    else:
                        # If no distance matrix of new points is made yet, then make into M_temp
                        if np.shape(new_states)[0] == 2:
                            M_temp = sp.spatial.distance_matrix(new_states[:, 0,:],new_states[:,0,:])
                            M_temp = np.multiply(M_temp, M_temp<epsilon)

                        # Test if a new clique average is able to decrease the overall number of triangles
                        simplice_positions = np.array([np.mean(states[clique,:,:], axis = 0)])
                        M_test = append_distance_matrix(M_temp, simplice_positions, new_states, epsilon)
                        if count_cliques_from_M(M_test)[2] == 0: #num_3_cliques
                            # New point is permissible, so copy over new distance matrix and append new point
                            new_states = np.append(new_states, simplice_positions, axis = 0)
                            M_temp = M_test

                        # Add redundant nodes to list for later deletion    
                        redundant_nodes = redundant_nodes + clique
                else:
                    continue
            except:
                break

        # Collect redundant nodes
        redundant_nodes_idx = list(set(redundant_nodes))

        # Delete redundant states from clique complex and append new states
        states = np.delete(states, redundant_nodes_idx, axis = 0)
        states = np.append(states, new_states, axis = 0)

        # Print Summary
        print(f'Removed all {k_target}-Cliques, Deleted Nodes: {len(redundant_nodes_idx)}, '+
            f'Added Nodes: {np.shape(new_states)[0]}, '+
            f'New Network Size: {np.shape(states)[0]}')

        # Recalculate Network cliques statistics for next iteration of loop
        M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
        M = np.multiply(M, M<epsilon)
        G = nx.from_numpy_array(M)


        clique_counts = count_cliques_from_M(M)
        num_cliques = np.sum(clique_counts)
    print("Completed Attractor Trim")
    return (states, M)


# Clustering Coefficient Based Trimming
def trim_attractor_network_clustering(states, epsilon = 0.8, min_clustering = 0.5):
    while True: # break condition

        # Calculate spatial distance and clustering coefficient
        M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
        M = np.multiply(M, M<epsilon)
        G = nx.from_numpy_array(M) # Get Distance graph
        clustering_coeff = np.array(list(nx.clustering(G).values()))
        node_degree = np.sum(M!=0, axis = 1)
        node_id = np.array(range(np.shape(M)[0]))

        # Filter out nodes with minimum clustering coefficient
        idx = np.where(np.array(clustering_coeff)>min_clustering)[0]
        clustering_coeff = clustering_coeff[idx]
        node_degree = node_degree[idx]
        node_id = node_id[idx]
        if len(node_degree) == 0: # Breaking condition i.e. system is already fully trimmed
            print(f"No new clusters found. Simplification complete.")
            break
        
        # Filter out removable nodes by taking only those with the maximum current degree
        max_degree = np.amax(node_degree)
        if max_degree <= 2: # Ignore edges
            print(f"No new clusters found. Simplification complete.")
            break

        idx = np.where(np.array(node_degree)==max_degree)[0]
        clustering_coeff = clustering_coeff[idx]
        node_degree = node_degree[idx]
        node_id = node_id[idx]

        print(f"Found {len(idx)} high cluster nodes, with maximal degree {max_degree}.")

        sorted_node_indices = np.flip(np.lexsort((clustering_coeff, node_degree)))

        # Create empty set to store nodes to be deleted
        redundant_nodes = set()
        new_states = np.zeros((0,2,np.shape(states)[2]))

        # Start Trimming nodes
        for id in sorted_node_indices:
            target_id = node_id[id]
            cluster_id = set(np.where(M[target_id,:]>0)[0].flatten()).union({target_id})


            # Check if neighbours are already in set
            n_intersections = len(redundant_nodes.intersection(cluster_id))

            if n_intersections < len(cluster_id)-1: # i.e. simplifying cluster into a point will decrease the point cloud
                # Then create new point in the data
                clique = np.array(list(cluster_id))
                weights = M[target_id, clique]
                
                # Get weighted average spatial position
                simplice_positions = np.array([np.average(states[clique,:,:], axis = 0, weights = weights)])
                new_states = np.append(new_states, simplice_positions, axis = 0)
                redundant_nodes = redundant_nodes.union(cluster_id)

        # Update new node states
        redundant_nodes_idx = list(set(redundant_nodes))
        # Delete redundant states from clique complex and append new states
        states = np.delete(states, redundant_nodes_idx, axis = 0)
        states = np.append(states, new_states, axis = 0)

        print(f'Removed all clusters of degree {max_degree}, Deleted Nodes: {len(redundant_nodes_idx)}, '+
            f'Added Nodes: {np.shape(new_states)[0]}, '+
            f'New Network Size: {np.shape(states)[0]}')

    #Recalculate distances
    M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
    M = np.multiply(M, M<epsilon)
        
    return (states, M)


# %% Extracted Connected components
"""
Extract union of largest connected components, filter states, and reform network
Uses EPSILON_FLOW rather than EPSILON
"""

def extract_connected_components(states, epsilon_flow = 0.05, component_min_size = 10):
    # Distance matrix
    M = sp.spatial.distance_matrix(states[:, 0,:],states[:,0,:])
    M = np.multiply(M, M<epsilon_flow)

    G = nx.from_numpy_array(M) # Get Distance graph

    largest_cc = np.array(list(max(nx.connected_components(G), key=len)))
    print("Size of Largest Component: "+repr(len(largest_cc)))

    component_node_list = []

    for component in sorted(nx.connected_components(G), key=len, reverse=True):
        if len(component)>component_min_size:
            component_node_list = component_node_list + list(component)

    component_node_list = list(set(component_node_list))

    print(f'Size of collective components with more than {component_min_size} nodes: {len(component_node_list)}')
        
    # Extract largest network
    # M_cc = M[largest_cc, :]
    # M_cc = M_cc[:, largest_cc]

    M_cc = M[component_node_list, :]
    M_cc = M_cc[:, component_node_list]

    # Remove unwanted states
    # states = states[largest_cc,:]
    output_states = states[component_node_list,:]

    return output_states

"""
Trims flow network by removing dead ends
"""
def trim_flow_network(states, epsilon_flow = 0.8, k_scale = 1):
    print("Trimming Flow Network...")
    # Calculates all the pairwise distances from a source node to all other points
    M_flow = sp.spatial.distance_matrix(states[:,1,:],states[:,0,:]) # This is almost diagonal, but not quite
    M_flow = np.multiply(np.exp(-k_scale*M_flow/epsilon_flow), M_flow<epsilon_flow)
    np.fill_diagonal(M_flow, 0) # Make diagonal entries 0 because shouldn't have self loops

    # Need to check for cases where there is no possible outgoing path (will lead to NaNs)
    vec = np.sum(M_flow, axis = 1) 
    dead_end_idx = np.where(vec==0)[0] # Node index that have no outgoing paths

    while len(dead_end_idx) > 0:
        vec = np.sum(M_flow, axis = 1) 
        dead_end_idx = np.where(vec==0)[0] # Node index that have no outgoing paths
        
        # Delete nodes in transition matrix
        M_flow = np.delete(M_flow, dead_end_idx, axis = 0)
        M_flow = np.delete(M_flow, dead_end_idx, axis = 1)

        # Delete nodes for next iteration of scanning
        vec = np.delete(vec, dead_end_idx, axis = 0)

        # Delete states associated with dead end nodes
        states = np.delete(states, dead_end_idx, axis = 0)
        print(f'Deleted Nodes: {len(dead_end_idx)}, Network Size: {np.shape(M_flow)[0]}')

    print("Completed Simplification")

    vec = np.sum(M_flow, axis = 1) 

    for i in range(np.shape(M_flow)[1]):
        M_flow[i,:] = M_flow[i,:]/np.sum(M_flow[i,:])

    return (states, M_flow)

"""
Same as trim_flow_network, but works directly from a direct adjacency matrix M_flow
originally calculated by fitting dynamics to attractor skeleton
Also trims states accordingly if nodes are deleted
"""
def trim_flow_network_from_M(states, M_flow):
    print("Trimming Flow Network...")

    # Need to check for cases where there is no possible outgoing path (will lead to NaNs)
    vec = np.sum(M_flow, axis = 1) 
    dead_end_idx = np.where(vec==0)[0] # Node index that have no outgoing paths

    while len(dead_end_idx) > 0:
        vec = np.sum(M_flow, axis = 1) 
        dead_end_idx = np.where(vec==0)[0] # Node index that have no outgoing paths
        
        # Delete nodes in transition matrix
        M_flow = np.delete(M_flow, dead_end_idx, axis = 0)
        M_flow = np.delete(M_flow, dead_end_idx, axis = 1)

        # Delete nodes for next iteration of scanning
        vec = np.delete(vec, dead_end_idx, axis = 0)

        # Delete states associated with dead end nodes
        states = np.delete(states, dead_end_idx, axis = 0)

        print(f'Deleted Nodes: {len(dead_end_idx)}, Network Size: {np.shape(M_flow)[0]}')

    print("Completed Simplification")

    vec = np.sum(M_flow, axis = 1) 

    for i in range(np.shape(M_flow)[1]):
        M_flow[i,:] = M_flow[i,:]/np.sum(M_flow[i,:])

    return (states, M_flow)

"""
Constructs the flow network. If a node has no outgoing paths, then rewire to the nearest available node
"""
def rewire_flow_network(states, epsilon_flow = 0.8, k_scale = 1):
    print("Rewiring Flow Network...")
    # Calculates all the pairwise distances from a source node to all other points
    M_flow_unfiltered = sp.spatial.distance_matrix(states[:,1,:],states[:,0,:]) # This is almost diagonal, but not quite
    M_flow = np.multiply(np.exp(-k_scale*M_flow_unfiltered/epsilon_flow), M_flow_unfiltered<epsilon_flow)
    np.fill_diagonal(M_flow, 0) # Make diagonal entries 0 because shouldn't have self loops

    # Need to check for cases where there is no possible outgoing path (will lead to NaNs)
    vec = np.sum(M_flow, axis = 1) 
    dead_end_idx = np.where(vec==0)[0] # Node index that have no outgoing paths

    # Store disparity between rewired distance and epsilon_flow
    dist_ratio = np.zeros(len(dead_end_idx))
    for i in range(len(dead_end_idx)):
        dead_node = dead_end_idx[i]
        dists = M_flow_unfiltered[dead_node,:] # Extract all node distances
        rewired_node_destination = np.argmin(dists)
        M_flow[dead_node, rewired_node_destination] = 1
        dist_ratio[i] = dists[rewired_node_destination]/epsilon_flow

    print(f"Completed Rewiring {len(dead_end_idx)} nodes. Rewired distance ratio: mean = {np.round(np.mean(dist_ratio), digits = 3)}, std = {np.round(np.std(dist_ratio),digits = 3)}")

    vec = np.sum(M_flow, axis = 1) 

    for i in range(np.shape(M_flow)[1]):
        M_flow[i,:] = M_flow[i,:]/np.sum(M_flow[i,:])

    return (states, M_flow)
