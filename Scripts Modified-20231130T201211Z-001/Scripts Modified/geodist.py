import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import connected_components
from sklearn.utils.graph import _fix_connected_components
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import sys

def floyd_warshall(adjacency_matrix):

    mat = np.asarray(adjacency_matrix)
    (nrows, ncols) = mat.shape
    assert nrows == ncols
    n = nrows
    for k in range(n):
        mat = np.minimum(mat, mat[np.newaxis,k,:] + mat[:,k,np.newaxis]) 
    return mat

def load_dataset(dataset_name):
    home = os.path.expanduser('~')
    dataset_path = os.path.join(home, 'Datasets', dataset_name)
    df = pd.read_csv(dataset_path, header=None)
    return df.to_numpy()[:, :-1]

def calc_pairwise_dists(arr):
    matrix = []
    for i, coords in enumerate(arr):
        coords = coords.reshape((1, -1))
        dists = np.linalg.norm(arr - coords, ord=2, axis=1)
        matrix.append(dists)

    return np.array(matrix)

def calc_nn(arr_dists, num_neighbor):
    idx = (np.argpartition(arr_dists, num_neighbor+1, axis=1))[:, :num_neighbor+1]
    weights = arr_dists[np.arange(arr_dists.shape[0])[:, None], idx]
    
    return idx, weights

def calc_weights(nearest_idxs, nearest_weights):
    weight_matrix = np.zeros((nearest_weights.shape[0], nearest_weights.shape[0]))
    weight_matrix[np.arange(weight_matrix.shape[0])[:, None], nearest_idxs] = nearest_weights
    return weight_matrix

def calc_geodesic_dists(weight_matrix):
    G = nx.from_numpy_matrix(np.matrix(weight_matrix), create_using=nx.Graph)
    length = dict(nx.all_pairs_bellman_ford_path_length(G))
    
    geodesic_weights = np.zeros(weight_matrix.shape)
    geodesic_weights[:, :] = -1
    
    for n in range(weight_matrix.shape[0]):
        for node in range(weight_matrix.shape[1]):
#             try:
            geodesic_weights[n, node] = length[n][node]
#             except KeyError:
#                 pass
#             print(f'{n*n}/{2000*2000}', end='\r')
    
    assert np.allclose(geodesic_weights, geodesic_weights.T, rtol=1e-05, atol=1e-08)
    
    maxv = geodesic_weights.max()
#     geodesic_weights[geodesic_weights == -1] = maxv * 1.5
    
    return geodesic_weights


#changed the function
def create_geodesic_matrix(X, n_neighbors):
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors)  
    nbrs_.fit(X)

    nbg = kneighbors_graph(
                nbrs_,n_neighbors=n_neighbors,
                mode="distance",
            )
    n_connected_components, labels = connected_components(nbg)
    print("connected", n_connected_components)
    #nbg = _fix_connected_components(
       #         X=nbrs_._fit_X,
       #         graph=nbg,
         #       n_connected_components=n_connected_components,
          #      component_labels=labels,
           #     mode="distance",
           # )
    X = nbrs_._fit_X
    nbg = _fix_connected_components(
                X=nbrs_._fit_X,
                graph=nbg,
                n_connected_components=n_connected_components,
                component_labels=labels,
                mode="distance",
            )
    n_connected_components, labels = connected_components(nbg)
            
    geod = shortest_path(nbg, directed = False)
    #print(np.max(geod),np.mean(geod))
    return geod

def create_euclidean_matrix(array):
    return calc_pairwise_dists(array)
#     samples = array.shape[0]
    

# def rescale_matrix(matrix, nmin=0.05, nmax=0.95, minv=None, maxv=None, image=False):
#     if image:
#         mask = np.eye(matrix.shape[0], dtype=bool)
#         mask = ~mask
#     else:
#         mask = np.ones(matrix.shape[0], dtype=bool)
    
#     if minv is None:
#         minv = matrix[mask].min()
    
#     if maxv is None:
#         maxv = matrix[mask].max()
        
#     if minv < 0:
#         matrix[mask] = matrix[mask] - minv
        
#     matrix[mask] = matrix[mask] * (nmax - nmin)
#     matrix[mask] = matrix[mask] / (maxv - minv)
#     matrix[mask] = matrix[mask] + nmin
    
#     return matrix, minv, maxv

def rescale_matrix(matrix, nmin=0.05, nmax=0.95, minv=None, maxv=None, image=False):
    if image:
        mask = np.eye(matrix.shape[0], dtype=bool)
        mask = ~mask
    else:
        mask = np.ones(matrix.shape[0], dtype=bool)
    
    if minv is None:
        minv = matrix[mask].min()
    
    if maxv is None:
        maxv = matrix[mask].max()
        
    matrix[mask] = matrix[mask] - minv
       
    matrix[mask] = matrix[mask] * (nmax - nmin)
    matrix[mask] = matrix[mask] / (maxv - minv)
    matrix[mask] = matrix[mask] + nmin

    return matrix, minv, maxv


def geodist_2(matrix, ep=5):
    Data = matrix
    [num_of_instance, dim]=Data.shape
    ed=euclidean_distances(Data)
    
    flag=0
#     ep=round(0.01*num_of_instance)
    if (ep<5):
        ep=5
    while(flag==0):
        indices=np.argsort(ed)
        M=np.sqrt(sys.float_info.max/num_of_instance)
        Adj=np.full(ed.shape,M)
        for i in range(num_of_instance):
            Adj[i,indices[i,1:ep+1]]=ed[i,indices[i,1:ep+1]]
            Adj[i,i]=0.
            
        for i in range(num_of_instance):
            for j in range(i+1,num_of_instance):
                if (Adj[i,j]!=M):
                    Adj[j,i]=Adj[i,j]
                else:
                    val=Adj[j,i]
                    Adj[i,j]=val
        
        Geodesic_Distance=floyd_warshall(Adj)
        return Geodesic_Distance