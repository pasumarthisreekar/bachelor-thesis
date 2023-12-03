This repo contains my bachelor's thesis, its supplementary material and the code for the autoencoder.

Formulation1log function in Regularizers_Geod: Contains the implementation of the proposed regularizer

create_geodesic_matrix function in geodist: Contains the implementation for calculating the geodesic distance on the original dataset
Note: In case of disconnected graphs while computing the nearest neighbor graph for the original dataset, for each pair of unconnected components, we compute all pairwise distances
    from one component to the other, and add a connection on the closest pair of samples.
