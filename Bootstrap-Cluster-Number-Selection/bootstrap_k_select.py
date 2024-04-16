import math
import numpy as np
import pandas as pd
from os import cpu_count
import multiprocessing as mp
from typing import Union, Literal
from types import FunctionType
import logging

# =================================== #
# >>> The Template Implementation <<< #
# =================================== #

def single_thread_boostrap_k_selection(
    X_data: pd.DataFrame,
    cluster_function: Union[callable, FunctionType],
    max_k_clusters: int = 10,
    num_bootstrap_samples: int = 100,
    n_cluster_argname: str = "n_components",
    with_corrections: bool = True,
    cluster_func_kwargs: dict = None
) -> int:
    '''
    Selection of the number of clusters via the bootstrap method as proposed
    by Fang and Wang.

    Algorithm description from Maslbeck and Wulff:
    1. Take bootstrap samples A & B from empirical dataset X_data
    2. Learn clusterings cluster_A & cluster_B from boostrap_sample_A and
       bootstrap_sample_B using some clustering algorithm
    3. Use clusterings cluster_A & cluster_B to compute assignments for each 
       object in original dataset X_data
    4. Use the assignments to compute the clustering distance 

    Args: 
        X_data (pd.DataFrame): The empirical observed dataset/distribution
        cluster_function (callable): A callable clustering algorithm/object
            that implements the fit and predict method.
        max_k_clusters (int): The max number of clusters, k, to consider
        num_bootstrap_samples (int): The number of bootstrap samples to 
            compute the instability. The higher this value, the more accurate
            this method is.
        n_cluster_argname (str): the argument keyword for cluster_function
            that defines the number of clusters to use in the algorithm. 
            For instance, for the KMeans in sklearn, the value of this 
            argument should be 'n_clusters'.
        with_corrections (bool): whether to use the correction Maslbeck & Wulff
        **cluster_func_kwargs: other keywords to include in cluster_function
    
    Return:
        best_k, cluster_dists: 
            The best number of clusters to use, and the dataframe containing
            all cluster instabilities for all bootstrap samples.
    '''
    best_ks = []
    cluster_dists = {}

    cluster_len = X_data.shape[0]
    # >>> Repeat process for B number of bootstrap samples <<< #
    for b in range(num_bootstrap_samples):
        prog_msg = f"Bootstrap sample {b + 1} of {num_bootstrap_samples}"

        # >>> Generate bootstrap sample pairs <<< #
        bootstrap_A = X_data.sample(frac = 1.0, 
                                    replace = True)
        bootstrap_B = X_data.sample(frac = 1.0, 
                                    replace = True)
        bootstrap_dists = []
        # >>> Compute cluster mapping for each number of clusters <<< #
        for num_clusters in range(2, max_k_clusters + 1):
            logging.info(f"{prog_msg}\tNum Clusters: {num_clusters}")
            # >>> Set number of clusters <<< #
            cluster_func_kwargs[n_cluster_argname] = num_clusters

            # >>> Apply clustering on bootstrap sample A <<< #
            cluster_A = cluster_function(
                **cluster_func_kwargs
            ).fit(bootstrap_A)

            # >>> Apply clustering on bootstrap sample B <<< #
            cluster_B = cluster_function(
                **cluster_func_kwargs
            ).fit(bootstrap_B)

            # >>> Apply clustering trained on bootstrap sample A & B
            #     on original data <<< #
            cluster_labels_A = cluster_A.predict(X_data)
            cluster_labels_B = cluster_B.predict(X_data)

            # >>> Generate Equality Matrix of Cluster Mappings <<< #
            indicator_A = np.tile(cluster_labels_A, (cluster_len, 1))
            equality_matrix_A = (indicator_A == indicator_A.T)

            indicator_B = np.tile(cluster_labels_B, (cluster_len, 1))
            equality_matrix_B = (indicator_B == indicator_B.T)

            # >>> Compute correction coefficients <<< #
            C1, C2 = maslbeck_wulff_correction(cluster_labels_A, 
                                               cluster_labels_B)

            # >>> Compute empirical distance <<< #
            empirical_dist = np.mean(equality_matrix_A != equality_matrix_B)
            corrected_distance = ((empirical_dist - C1) / (2 * C2)
                                  if with_corrections 
                                  else empirical_dist)
            
            # >>> Save Cluster Instability <<< #
            key = f"k={num_clusters}"
            cluster_dists[key] = (cluster_dists.get(key, [])
                                  + [corrected_distance])
            bootstrap_dists.append(corrected_distance)
        best_ks.append(np.argmin(bootstrap_dists) + 2)
    
    # Convert cluster instabilies into dataframe
    cluster_dists = pd.DataFrame(cluster_dists)

    return best_ks, cluster_dists

def maslbeck_wulff_correction(
    cluster_labels_A: np.ndarray,
    cluster_labels_B: np.ndarray
) -> tuple[float, float]:
    '''
    Correction terms to the selection of number of clusters via the 
    bootstrap method as proposed by Maslbeck and Wulff.

    As per their publication: 
    c1 = E(I_a)E(1 - I_b) + E(I_b)E(1 - I_a)
    c2 = sqrt(var(1 - I_a)) x sqrt(var(I_b))

    where, 
    var(I_s) = var(1 - I_s) = E(I_s)E(1 - I_s)
    E(I_s) = \sum{mC2} / nC2, m = size of cluster i for 1 <= i <= k

    var(I_a) = var(1 - I_a),
    var(I_b) = var(1 - I_b),
    E(I_a) = \sum{mC2} / nC2 for cluster mapping on bootstrap sample A
    E(I_b) = \sum{mC2} / nC2 for cluster mapping on bootstrap sample B

    Args:
        cluster_labels_A (list): list of cluster map labels for cluster
            function trained on bootstrap sample A
        
        cluster_labels_B (list): list of cluster map labels for cluster
            function trained on bootstrap sample B
    
    Returns: 
        C1, C2: the computed correction terms 
    '''

    # Lists to save correction terms of mC2 as per Maslbeck & Wulff
    cluster_pairing_size_A = []
    cluster_pairing_size_B = []
    
    cluster_n = len(cluster_labels_A)
    # Calculate the influence of the size of each cluster
    for cluster_label in np.unique(cluster_labels_A):
        cluster_size_A = np.sum(cluster_labels_A == cluster_label)
        cluster_size_B = np.sum(cluster_labels_B == cluster_label)
        # cluster size choose 2
        k_pair_A = math.comb(cluster_size_A, 2) 
        k_pair_B = math.comb(cluster_size_B, 2)

        cluster_pairing_size_A.append(k_pair_A)
        cluster_pairing_size_B.append(k_pair_B)

    # Compute correction terms as per Maslbeck & Wulff
    E_IA = np.sum(cluster_pairing_size_A) / math.comb(cluster_n, 2)
    E_IB = np.sum(cluster_pairing_size_B) / math.comb(cluster_n, 2)

    C1 = E_IA * (1 - E_IB) + (1 - E_IA) * E_IB
    C2 = np.sqrt(E_IA * (1 - E_IA)) * np.sqrt(E_IB * (1 - E_IB))

    return [C1, C2]

def compute_cluster_instability( # Refactor for only one k value
    X_data: pd.DataFrame,
    cluster_function: Union[callable, FunctionType],
    k_clusters: int,
    n_cluster_argname: str = "n_components",
    with_corrections: bool = True,
    cluster_func_kwargs: dict = None
) -> tuple[int, pd.Series]:
    '''Singular function that will applied to each k and bootstrap replicate.

    This is the function that is applied using multiprocessing.Pool.starmap.

    Args:
        X_data (pd.DataFrame): Input data for unsupervised clustering
        cluster_function (callable): Clustering class or callable that 
            implements both 'fit' and 'predict', and takes in 'k' as
            a hyperparameter. 
        k_clusters (int): the number of clusters, k, to use in this replicate.
        n_cluster_argname (str): The keyword arg name of cluster_function
            which identifies the k parameter
        with_corrections (bool): Whether to use Maslbeck Wulff correction 
        **cluster_func_kwargs: Additional keyword arguments to pass to 
            cluster_function
    Returns:
        Cluster instability distance. 
    
    '''
        
    cluster_len = X_data.shape[0]

    # >>> Generate two bootstrap samples <<< #
    bootstrap_sample_A = X_data.sample(frac = 1, replace = True)
    bootstrap_sample_B = X_data.sample(frac = 1, replace = True)

    cluster_dists = {}

    # >>> Compute cluster mapping for each number of clusters <<< #
    #for num_clusters in range(2, max_k_clusters + 1):
    # >>> Set number of clusters <<< #
    cluster_func_kwargs[n_cluster_argname] = k_clusters

    # >>> Apply clustering on bootstrap sample A <<< #
    cluster_A = cluster_function(
        **cluster_func_kwargs
    ).fit(bootstrap_sample_A)

    # >>> Apply clustering on bootstrap sample B <<< #
    cluster_B = cluster_function(
        **cluster_func_kwargs
    ).fit(bootstrap_sample_B)

    # >>> Apply clustering trained on bootstrap sample A & B
    #     on original data <<< #
    cluster_labels_A = cluster_A.predict(X_data).astype('uint8')
    cluster_labels_B = cluster_B.predict(X_data).astype('uint8')

    # >>> Generate Equality Matrix of Cluster Mappings <<< #
    indicator_A = np.tile(cluster_labels_A, (cluster_len, 1))
    equality_matrix_A = (indicator_A == indicator_A.T)

    indicator_B = np.tile(cluster_labels_B, (cluster_len, 1))
    equality_matrix_B = (indicator_B == indicator_B.T)

    # >>> Compute correction coefficients <<< #
    C1, C2 = maslbeck_wulff_correction(cluster_labels_A, cluster_labels_B)

    # >>> Compute empirical distance <<< #
    empirical_dist = np.mean(
        np.uint8(equality_matrix_A != equality_matrix_B)
    )
    corrected_distance = ((empirical_dist - C1) / (2 * C2)
                            if with_corrections 
                            else empirical_dist)
    
    # >>> Save Cluster Instability <<< #
    key = f"k={k_clusters}"
    cluster_dists[key] = corrected_distance

    # best_k = np.argmin(cluster_dists) + 2 # Best k for this bootstrap iteration
    logging.info(f"Completed {key} bootstrap iteration.")

    return cluster_dists

def parallelized_bootstrap_k_selection(
    X_data: pd.DataFrame,
    cluster_function: Union[callable, FunctionType],
    max_k_clusters: int = 10,
    num_bootstrap_samples: int = 100,
    n_cluster_argname: str = "n_components",
    with_corrections: bool = True,
    **cluster_func_kwargs
) -> tuple[int, pd.DataFrame]:
    '''Parallelized implementation of bootstrap k selection.

    Args:
        X_data (pd.DataFrame): Input data for unsupervised clustering
        cluster_function (callable): Clustering class or callable that 
            implements both 'fit' and 'predict', and takes in 'k' as
            a hyperparameter. 
        max_k_clusters (int): The maximum k value to consider
        num_bootstrap_samples (int): Number of bootstrap sample pairs, or 
            replicates to generate. 
        n_cluster_argname (str): The keyword arg name of cluster_function
            which identifies the k parameter
        with_corrections (bool): Whether to use Maslbeck Wulff correction 
        **cluster_func_kwargs: Additional keyword arguments to pass to 
            cluster_function
    Returns:
        The k values with the lowest instability in each bootstrap replicate. 
        The instabilities for each k value for each bootstrap sample. 
    '''

    mp_logger = mp.get_logger()
    mp_logger.setLevel(logging.INFO)
    
    # >>> Make arguments for parallelization <<< #
    mp_arg_iter = [
        [X_data, cluster_function, ((idx) % (max_k_clusters - 1) + 2), 
         n_cluster_argname, with_corrections, cluster_func_kwargs]
         for idx in range(int(num_bootstrap_samples * (max_k_clusters - 1)))
    ]

    # >>> Multiprocessing for parallelization <<< #
    compute_units = cpu_count() - 2
    with mp.Pool(processes = compute_units) as mp_pool:
        bootstrap_results = (
            mp_pool.starmap(
                compute_cluster_instability, 
                mp_arg_iter, 
                chunksize = compute_units
            )
        )
    # Get results from all bootstrap permutations
    instabilities = {}
    for k_instability in bootstrap_results:
        k_key, k_val = list(k_instability.items())[0]
        instabilities[k_key] = instabilities.get(k_key, []) + [k_val] 
    try:
        bootstrap_selection_results = pd.DataFrame(instabilities)

        # Combine the clustering instability results for all bootstrap samples
        # bootstrap_selection_results = pd.concat(instabilities, axis = 1).T
        # Replace inf and -inf values
        bootstrap_selection_results.replace([np.inf, -np.inf], np.nan, 
                                            inplace = True)
        best_ks = bootstrap_selection_results.idxmin(axis = 1)
        best_ks = best_ks.apply(lambda x: int(x.split("=")[-1]))

        return best_ks, bootstrap_selection_results
    except:
        return instabilities

def bootstrap_k_selection(
    X_data: pd.DataFrame, 
    cluster_function: Union[callable, FunctionType],
    max_k_clusters: int = 10,
    num_bootstrap_samples: int = 100,
    n_cluster_argname: str = "n_components",
    with_corrections: bool = True,
    sample_size_limit: int = 20_000,
    implementation: Literal['single', 'parallel'] = 'parallel',
    **cluster_func_kwargs
) -> tuple[list, pd.DataFrame]: 
    '''
    Args: 
        X_data (pd.DataFrame): The empirical observed dataset/distribution
        cluster_function (callable): A callable clustering algorithm/object
            that implements the fit and predict method.
        max_k_clusters (int): The max number of clusters, k, to consider
        num_bootstrap_samples (int): The number of bootstrap samples to 
            compute the instability. The higher this value, the more accurate
            this method is.
        n_cluster_argname (str): the argument keyword for cluster_function
            that defines the number of clusters to use in the algorithm. 
            For instance, for the KMeans in sklearn, the value of this 
            argument should be 'n_clusters'.
        with_corrections (bool): whether to use the correction Maslbeck & Wulff
        sample_size_limit (int): The maximum sample size - take a sample if 
            X_data is larger than this value. Avoid slow computation or 
            crashes due to extremely high volume data.
        implementation (Literal): 'parallel' for implementation using 
            multiprocessing.Pool, 'single' for single-threaded implementation.
        **cluster_func_kwargs: other keywords to include in cluster_function
    
    Return:
        best_k, cluster_dists: 
            The best number of clusters to use, and the dataframe containing
            all cluster instabilities for all bootstrap samples.
    '''
    if implementation == 'parallel':
        boot_k_select_fn = parallelized_bootstrap_k_selection
    elif implementation == 'single':
        boot_k_select_fn = single_thread_boostrap_k_selection
    else:
        raise ValueError("implementation must be 'single' for single thread "
                         "or 'parallel' for multiprocessing")
    X_data = X_data.sample(n = min(X_data.shape[0], sample_size_limit))

    return boot_k_select_fn(
        X_data = X_data, 
        cluster_function = cluster_function, 
        max_k_clusters = max_k_clusters,
        num_bootstrap_samples = num_bootstrap_samples,
        n_cluster_argname = n_cluster_argname, 
        with_corrections = with_corrections,
        **cluster_func_kwargs
    )