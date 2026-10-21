import numpy as np
import cupy as cp
import h5py
from tqdm import tqdm
from cupyx.scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist as npcdist
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata
import psutil
import numba as nb

### METHOD TO SAVE GRAPHS ###

def saveGraph(n, graph, outfile):
    print("Building adjacency matrix...")
    E = []
    for i in range(n):
        for j in range(1, len(graph[i])):
            E.append((graph[i][0], graph[i][j]))

    # Create adjacency matrix on GPU
    # adj = cp.zeros((n, n), dtype=cp.int16)
    # for (u, v) in E:
    #    adj[u][v] = 1

    # Transfer back to CPU and save
    # adj_cpu = cp.stack(adj.nonzero()).get()
    np.save(outfile, np.array(E))

    print(f"Saved adjacency matrix with {len(E)} edges. Shape: {adj_cpu.shape}")

########################
### GREEDY SET COVER ###

def buildSetsOfSource(permutation_matrix, source):
    """
    GPU-accelerated version - vectorized computation.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
        source: int, source vertex index
    
    Returns:
        CuPy array (n, n) on GPU
    """
    # Vectorized comparison: all rows compared to source row at once
    threshold = permutation_matrix[:, source][:, None]
    setsFromSourceVia = (permutation_matrix < threshold).astype(cp.uint16)
    
    return setsFromSourceVia

def greedySetCover(permutation_matrix, source):
    """
    GPU-accelerated greedy set cover.

    Precomputes sets as bool (4.9 GB for n=70k) once per source, then uses
    a matmul for scoring each iteration.  Peak VRAM during the matmul:
      perm (9.8 GB) + sets_bool (4.9 GB) + int32 matmul temp (14.4 GB) = 29.1 GB.

    Args:
        permutation_matrix: CuPy array (n, n) uint16 on GPU
        source: int, source vertex index

    Returns:
        list of (neighbor_id, uncov_count) tuples (transferred to CPU)
    """
    n = permutation_matrix.shape[0]
    covered = cp.zeros(n, dtype=cp.bool_)
    covered[source] = True
    edges = []
    uncovered = n - 1

    # Precompute sets as float32 once — fixed for this source.
    # Casting once before the loop avoids a 14.4 GB dynamic allocation inside
    # each matmul iteration (which corrupts GPU state via cuBLAS int32 path).
    # float32 represents integers exactly up to 2^24 >> n, so no precision loss.
    threshold = permutation_matrix[:, source]
    sets_f32 = (permutation_matrix < threshold[:, None]).astype(cp.float32)  # (n, n) float32, 14.4 GB

    while uncovered > 0:
        scores = (~covered).astype(cp.float32) @ sets_f32   # standard cublasSgemm

        index = int(cp.argmax(scores))

        newly_covered = (sets_f32[:, index] != 0) & (~covered)
        uncovered -= int(cp.sum(newly_covered))
        covered |= newly_covered
        edges.append((index, uncovered))

    return edges

def buildSetCoverGraph(permutation_matrix):
    """
    Build graph using GPU-accelerated operations.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
    
    Returns:
        list of edge lists for each vertex
    """
    n = permutation_matrix.shape[0]
    edgeSet = []
    
    print("Building graph on GPU")

    for source in tqdm(range(n)):
        edges = greedySetCover(permutation_matrix, source)
        edgeSet.append(edges)

    return edgeSet

#######################

####################
### ROBUST PRUNE ###

def robustPrune(source, permutation_matrix):
    """
    GPU-optimized version of robustPrune using CuPy.
    
    Args:
        source: Source vertex index
        permutation_matrix: Distance matrix (cupy.ndarray on GPU)
    
    Returns:
        List of pruned edges in order
    """
    n = permutation_matrix.shape[0]
    
    # Get distances from source to all vertices
    dist_from_source = permutation_matrix[source].copy()
    
    # Create mask for active vertices (exclude source)
    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = []
    
    while cp.any(active):
        # Find closest active vertex
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()
        
        # Add to edges and deactivate
        edges.append(waypoint)
        active[waypoint] = False
        
        # Vectorized pruning: remove vertices closer to waypoint than to source
        # For all active vertices, check if dist(waypoint, v) <= dist(source, v)
        prune_mask = (permutation_matrix[:, waypoint] <= permutation_matrix[:, source]) & active
        active[prune_mask] = False
    
    return edges

def memEfficientRobustPrune(source, dataset):
    n = dataset.shape[0]
    dist_from_source = cdist(dataset[source:source+1], dataset, metric='euclidean').flatten()

    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (cdist(dataset, dataset[waypoint:waypoint+1], metric='euclidean').ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges

##################################
### BATCHED SINGLE-MACHINE PRUNE ###
# GPU (CuPy) and CPU (NumPy/BLAS) variants share the same algorithm.
# The CPU path relies on NumPy's BLAS backend (OpenBLAS / MKL) for
# multi-threaded matmuls; set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
# in the job script to control core count.

def precomputeAugMatrix(dataset):
    """
    Build the augmented dataset matrix for fast squared-Euclidean distances via matmul.

    The identity  ||v - x||^2 = [-2v | ||v||^2 | 1] @ [x | 1 | ||x||^2]^T
    lets a batch of B queries compute B×n squared distances with one (B, d+2)@(n, d+2).T
    matmul — no cdist loop, full GPU tensor-core utilization.

    Call once per dataset and pass the result to batchedEuclideanRobustPrune.
    """
    n, d = dataset.shape
    data_f32 = dataset.astype(cp.float32)
    sq_norms = cp.einsum('ij,ij->i', data_f32, data_f32)   # (n,)  avoids n×d temp
    X_aug = cp.empty((n, d + 2), dtype=cp.float32)
    X_aug[:, :d]  = data_f32
    X_aug[:, d]   = 1.0
    X_aug[:, d+1] = sq_norms
    return X_aug


def _query_aug(vecs, d):
    """Build augmented query matrix: (k, d) → (k, d+2) for the matmul trick."""
    sq = cp.einsum('ij,ij->i', vecs, vecs)   # (k,)
    V = cp.empty((vecs.shape[0], d + 2), dtype=cp.float32)
    V[:, :d]  = -2.0 * vecs
    V[:, d]   = sq
    V[:, d+1] = 1.0
    return V


def batchedEuclideanRobustPrune(sources, dataset, X_aug, sparse_threshold=2_000_000,
                                alpha=1.0):
    """
    Compute navigable-graph neighborhoods for a batch of source nodes in parallel.

    Each round all active sources simultaneously:
      1. Select their next edge via a single vectorized argmin across uncovered points.
      2. Share a batched matmul to compute waypoint→uncovered distances.
         When the union of uncovered sets across all active sources shrinks below
         sparse_threshold, the matmul targets only that subset (|union| << n cols).
      3. Update uncov_mask for every source at once.

    Args:
        sources:          list of int, source indices (length B).
        dataset:          cp.ndarray (n, d), float32, already on GPU.
        X_aug:            precomputeAugMatrix(dataset) result, shape (n, d+2).
        sparse_threshold: switch to sub-matmul when union of uncovered < this size.
        alpha:            alpha-reachability parameter, >= 1. Waypoint w covers point p
                          only when d(w, p) < d(source, p) / alpha, so larger alpha
                          demands strictly more progress per edge and yields denser
                          neighborhoods. alpha = 1 is the original coverage rule.
                          Distances here are *squared* euclidean, so the comparison is
                          applied as d2(w, p) < d2(source, p) / alpha^2.

    Returns:
        list of B neighborhoods; each neighborhood is a list of
        (neighbor_id: int, uncov_count: int) tuples, where uncov_count is the
        size of the uncovered set *after* that edge takes effect. So tuple j
        reports what remains uncovered once the first j+1 edges are in place.
    """
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1, got {alpha}")

    n, d = dataset.shape
    B = len(sources)
    sources_cp = cp.array(sources, dtype=cp.int64)

    # Distances below are squared euclidean, so the alpha-reachability test
    # d(w, p) < d(source, p) / alpha  becomes  d2(w, p) < d2(source, p) / alpha^2,
    # applied as  d2(w, p) * alpha^2 < d2(source, p)  at each comparison. Prescaling
    # dists_matrix by 1/alpha^2 instead would save the per-round multiply, but it
    # shrinks every source distance toward zero in float32 and the distances that
    # decide late rounds are already small, so the multiply stays on the fresh side.
    alpha_sq = float(alpha) ** 2

    # INIT — one (B, n) matmul covers all sources at once
    sv = dataset[sources_cp].astype(cp.float32)         # (B, d)
    dists_matrix = _query_aug(sv, d) @ X_aug.T          # (B, n) squared euclidean
    cp.maximum(dists_matrix, 0.0, out=dists_matrix)     # clamp floating-point noise

    # uncov_mask[i, j] = True  iff  point j is still uncovered for source i
    uncov_mask = cp.ones((B, n), dtype=cp.bool_)
    uncov_mask[cp.arange(B), sources_cp] = False        # sources don't cover themselves

    neighborhoods = [[] for _ in range(B)]
    active = list(range(B))     # indices into [0, B) still running

    while active:
        act = cp.array(active, dtype=cp.int64)           # (k,)

        # Gather rows for active sources (fancy index → copy, intentional)
        act_uncov = uncov_mask[act]                      # (k, n)
        act_dists = dists_matrix[act]                    # (k, n)

        # Vectorized waypoint selection: argmin over uncovered points for all active sources
        masked = cp.where(act_uncov, act_dists, cp.inf)  # (k, n)
        waypoints = cp.argmin(masked, axis=1)             # (k,)
        waypoints_np = waypoints.get()

        # Mark chosen waypoints as covered in the local copy
        act_uncov[cp.arange(len(active)), waypoints] = False

        # Union of uncovered points across all active sources — drives sparse vs dense
        union_mask = cp.any(act_uncov, axis=0)           # (n,)
        union_size = int(cp.sum(union_mask))

        if union_size > 0:
            wp_vecs = dataset[waypoints].astype(cp.float32)   # (k, d)
            W_aug   = _query_aug(wp_vecs, d)                  # (k, d+2)

            # alpha^2 scales the waypoint distances (see alpha_sq above). Folding it
            # into the matmul result keeps act_dists at full float32 magnitude.
            if union_size < sparse_threshold:
                # Sparse: matmul only against the uncovered subset of columns
                u_idx = cp.where(union_mask)[0]               # (u,)
                D_sub = alpha_sq * (W_aug @ X_aug[u_idx].T)   # (k, u)
                cp.maximum(D_sub, 0.0, out=D_sub)
                prune = D_sub < act_dists[:, u_idx]           # (k, u)
                sub   = act_uncov[:, u_idx]                   # (k, u) copy
                act_uncov[:, u_idx] = sub & ~prune            # scatter back
            else:
                # Dense: full matmul (union too large to gain from sparsity)
                D = alpha_sq * (W_aug @ X_aug.T)              # (k, n)
                cp.maximum(D, 0.0, out=D)
                act_uncov &= ~(D < act_dists)

        # Write modified rows back into uncov_mask
        uncov_mask[act] = act_uncov

        # Record (neighbor_id, uncov_count) with the uncovered count *after* this
        # edge takes effect — i.e. after both marking the waypoint covered and
        # applying the pruning above. Consumers (coverage_to_degree_analysis.py,
        # edge_to_coverage_analysis.py) read tuple j as "uncov remaining once the
        # first j+1 edges are in place"; distributed_robust_prune.py defers its
        # append for the same reason.
        uncov_counts = cp.sum(act_uncov, axis=1).get()   # (k,) on CPU
        for li, gi in enumerate(active):
            neighborhoods[gi].append((int(waypoints_np[li]), int(uncov_counts[li])))

        # Drop sources whose uncovered set is now empty
        still_active = cp.any(act_uncov, axis=1).get()   # (k,) bool
        active = [gi for li, gi in enumerate(active) if still_active[li]]

    return neighborhoods

####################################
### CPU VARIANTS (NumPy / BLAS) ###

def precomputeAugMatrixCPU(dataset):
    """CPU counterpart of precomputeAugMatrix — identical logic, NumPy arrays."""
    n, d = dataset.shape
    data_f32 = dataset.astype(np.float32)
    sq_norms = np.einsum('ij,ij->i', data_f32, data_f32)   # (n,)
    X_aug = np.empty((n, d + 2), dtype=np.float32)
    X_aug[:, :d]  = data_f32
    X_aug[:, d]   = 1.0
    X_aug[:, d+1] = sq_norms
    return X_aug


def _query_aug_cpu(vecs, d):
    """CPU counterpart of _query_aug."""
    sq = np.einsum('ij,ij->i', vecs, vecs)   # (k,)
    V = np.empty((vecs.shape[0], d + 2), dtype=np.float32)
    V[:, :d]  = -2.0 * vecs
    V[:, d]   = sq
    V[:, d+1] = 1.0
    return V


def batchedEuclideanRobustPruneCPU(sources, dataset, X_aug, sparse_threshold=2_000_000,
                                   alpha=1.0):
    """
    CPU counterpart of batchedEuclideanRobustPrune.

    Uses NumPy matmuls backed by OpenBLAS/MKL, which parallelise across all
    allocated cores automatically (controlled by OMP_NUM_THREADS /
    OPENBLAS_NUM_THREADS in the job script).  No .get() transfers needed.

    Args / returns: identical to batchedEuclideanRobustPrune, alpha included.
    """
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1, got {alpha}")

    n, d = dataset.shape
    B = len(sources)
    sources_arr = np.array(sources, dtype=np.int64)

    # Squared distances — see the GPU variant for why alpha^2 multiplies the
    # waypoint side rather than dividing the source side.
    alpha_sq = float(alpha) ** 2

    # INIT — one (B, n) BLAS matmul
    sv = dataset[sources_arr].astype(np.float32)         # (B, d)
    dists_matrix = _query_aug_cpu(sv, d) @ X_aug.T      # (B, n) squared euclidean
    np.maximum(dists_matrix, 0.0, out=dists_matrix)

    uncov_mask = np.ones((B, n), dtype=np.bool_)
    uncov_mask[np.arange(B), sources_arr] = False

    neighborhoods = [[] for _ in range(B)]
    active = list(range(B))

    while active:
        act = np.array(active, dtype=np.int64)           # (k,)

        act_uncov = uncov_mask[act]                      # (k, n) copy
        act_dists = dists_matrix[act]                    # (k, n) copy

        masked = np.where(act_uncov, act_dists, np.inf)  # (k, n)
        waypoints = np.argmin(masked, axis=1)             # (k,)

        act_uncov[np.arange(len(active)), waypoints] = False

        union_mask = np.any(act_uncov, axis=0)           # (n,)
        union_size = int(np.sum(union_mask))

        if union_size > 0:
            wp_vecs = dataset[waypoints].astype(np.float32)   # (k, d)
            W_aug   = _query_aug_cpu(wp_vecs, d)              # (k, d+2)

            # alpha^2 folded into the matmul result — see the GPU variant.
            if union_size < sparse_threshold:
                u_idx = np.where(union_mask)[0]               # (u,)
                D_sub = alpha_sq * (W_aug @ X_aug[u_idx].T)   # (k, u)
                np.maximum(D_sub, 0.0, out=D_sub)
                prune = D_sub < act_dists[:, u_idx]           # (k, u)
                sub   = act_uncov[:, u_idx]                   # (k, u) copy
                act_uncov[:, u_idx] = sub & ~prune            # scatter back
            else:
                D = alpha_sq * (W_aug @ X_aug.T)              # (k, n)
                np.maximum(D, 0.0, out=D)
                act_uncov &= ~(D < act_dists)

        uncov_mask[act] = act_uncov

        # uncov count *after* this edge takes effect — see the GPU variant.
        uncov_counts = np.sum(act_uncov, axis=1)         # (k,)
        for li, gi in enumerate(active):
            neighborhoods[gi].append((int(waypoints[li]), int(uncov_counts[li])))

        still_active = np.any(act_uncov, axis=1)         # (k,) — already on CPU
        active = [gi for li, gi in enumerate(active) if still_active[li]]

    return neighborhoods

####################################

def load_hdf5_safe(filepath, dataset_name='data'):
    """Load HDF5 with memory availability check"""
    
    with h5py.File(filepath, 'r') as f:
        dset = f[dataset_name]
        required_gb = dset.nbytes / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"Dataset size: {required_gb:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")
        
        if required_gb > available_gb * 0.8:  # Keep 20% buffer
            raise MemoryError(
                f"Not enough RAM! Need {required_gb:.2f} GB, "
                f"have {available_gb:.2f} GB available"
            )
        
        print("Loading into RAM...")
        data = dset[:]
        print("Load complete!")
        
    return data

# @nb.njit(parallel=True, fastmath=True)
# def sqeuclidean_distances_numba(point, dataset):
#     """
#     Blazingly fast squared Euclidean distances
#     """
#     n_samples = dataset.shape[0]
#     n_features = dataset.shape[1]
#     distances = np.empty(n_samples, dtype=np.float32)
    
#     for i in nb.prange(n_samples):
#         dist = 0.0
#         for j in range(n_features):
#             diff = dataset[i, j] - point[j]
#             dist += diff * diff
#         distances[i] = dist
    
#     return distances

# def cpu_memEfficientRobustPrune(source, dataset):
#     n = dataset.shape[0]
#     dist_from_source = sqeuclidean_distances_numba(dataset[source], dataset)
#     # dist_from_source = npcdist(dataset[source], dataset, metric='sqeuclidean').flatten()

#     active = np.ones(n, dtype=np.bool_)
#     active[source] = False
    
#     edges = [source]

#     while np.any(active):
#         print(f"Degree: {len(edges) - 1}, Left: {np.sum(active)}")
#         masked_dist = np.where(active, dist_from_source, np.inf)
#         waypoint = np.argmin(masked_dist).item()

#         edges.append(waypoint)
#         active[waypoint] = False
        
#         # print(active.shape, dist_from_source.shape)

#         prune_mask = (sqeuclidean_distances_numba(dataset[waypoint], dataset) < dist_from_source) & active
#         active[prune_mask] = False

#     return edges

@nb.njit(parallel=True, fastmath=True)
def sqeuclidean_distances_numba(point, dataset):
    """Blazingly fast squared Euclidean distances"""
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]
    distances = np.empty(n_samples, dtype=np.float32)
    
    for i in nb.prange(n_samples):
        dist = 0.0
        for j in range(n_features):
            diff = dataset[i, j] - point[j]
            dist += diff * diff
        distances[i] = dist
    
    return distances

@nb.njit(parallel=True, fastmath=True)
def sqeuclidean_distances_numba_inplace(point, dataset, out):
    """Compute distances in-place to avoid allocation"""
    n_samples = dataset.shape[0]
    n_features = dataset.shape[1]
    
    for i in nb.prange(n_samples):
        dist = 0.0
        for j in range(n_features):
            diff = dataset[i, j] - point[j]
            dist += diff * diff
        out[i] = dist

def cpu_memEfficientRobustPrune(source, dataset):
    n = dataset.shape[0]
    
    # Use float32 for all distance arrays (half memory vs float64)
    dist_from_source = sqeuclidean_distances_numba(dataset[source], dataset)
    
    # Use uint8 for boolean mask (1 byte vs 8 bytes per element)
    active = np.ones(n, dtype=np.uint8)
    active[source] = 0
    
    # Pre-allocate arrays to avoid repeated allocations
    edges = np.empty(n, dtype=np.int32)  # Use int32 instead of Python list
    edges[0] = source
    edge_count = 1
    
    # Pre-allocate distance buffer (reuse for each waypoint)
    dist_buffer = np.empty(n, dtype=np.float32)
    
    # Track active count to avoid np.sum calls
    active_count = n - 1
    
    while active_count > 0:
        print(f"Degree: {edge_count - 1}, Left: {active_count}")
        
        # Find minimum among active points (avoid np.where with np.inf)
        active_indices = np.flatnonzero(active)
        waypoint = active_indices[dist_from_source[active_indices].argmin()]
        
        edges[edge_count] = waypoint
        edge_count += 1
        
        active[waypoint] = 0
        active_count -= 1
        
        if active_count == 0:
            break
        
        # Reuse pre-allocated buffer instead of creating new array
        sqeuclidean_distances_numba_inplace(dataset[waypoint], dataset, dist_buffer)
        
        # Prune in-place
        prune_mask = (dist_buffer < dist_from_source) & active.astype(np.bool_)
        prune_count = np.count_nonzero(prune_mask)
        
        if prune_count > 0:
            active[prune_mask] = 0
            active_count -= prune_count
    
    return edges[:edge_count].tolist()

def angularRobustPrune(source, dataset):
    n = dataset.shape[0]

    cosine_distance = cdist(dataset[source:source+1], dataset, metric='cosine').flatten()
    dist_from_source = cp.arccos(cosine_distance)
    
    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (cp.arccos(cdist(dataset, dataset[waypoint:waypoint+1], metric='cosine')).ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges

def jaccardRobustPrune(source, dataset):
    n = dataset.shape[0]

    dist_from_source = npcdist(dataset[source], dataset, metric='jaccard').flatten()
    
    active = np.ones(n, dtype=np.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = np.where(active, dist_from_source, np.inf)
        waypoint = np.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (npcdist(dataset, dataset[waypoint], metric='jaccard').ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges

def memBuildRobustPruneGraph(dataset):
    n = dataset.shape[0]
    edgeSet = []

    for source in tqdm(range(n)):
        edges = memEfficientRobustPrune(source, dataset)
        edgeSet.append(edges)

    return edgeSet

def buildRobustPruneGraph(permutation_matrix):
    """
    Build graph using GPU-accelerated operations.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
    
    Returns:
        list of edge lists for each vertex
    """
    n = permutation_matrix.shape[0]
    edgeSet = []
    
    print("Building graph on GPU")
    for source in tqdm(range(n)):
        edges = robustPrune(source, permutation_matrix)
        edgeSet.append(edges)
    
    return edgeSet

########################

########################
# BILLION SCALE ON GPU #
########################
def compute_distances_batched_GPU(query_vec, dataset, batch_size=100000):
    n = dataset.shape[0]
    distances = cp.empty(n, dtype=cp.float32)
    
    for start in tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch = cp.array(dataset[start:end], dtype=cp.float32)
        distances[start:end] = cdist(query_vec, batch, metric='sqeuclidean')[0]
    
    return distances


def compute_distances_batched_indexed_GPU(query_vec, dataset, indices_np, batch_size=100000):
    """Compute distances to specific indices in the dataset."""
    n = len(indices_np)
    distances = cp.empty(n, dtype=cp.float32)
    for start in tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_indices = indices_np[start:end]
        batch = cp.array(dataset[batch_indices], dtype=cp.float32)
        distances[start:end] = cdist(query_vec, batch, metric='sqeuclidean')[0]
    return distances


def billionRobustPrune_GPU(source, dataset, batch_size=100000):
    n = dataset.shape[0]
    source_vec = cp.array(dataset[source:source+1], dtype=cp.float32)
    dist_from_source = compute_distances_batched(source_vec, dataset, batch_size)
    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    edges = [source]
    while cp.any(active):
        print(f"{cp.sum(active):,} left")
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()
        edges.append(waypoint)
        active[waypoint] = False
        # Get indices of active points and convert to NumPy for h5py indexing
        active_indices = cp.where(active)[0]
        active_indices_np = active_indices.get()
        
        waypoint_vec = cp.array(dataset[waypoint:waypoint+1], dtype=cp.float32)
        # Compute distances only to active points (batched)
        dist_from_waypoint_active = compute_distances_batched_indexed(
            waypoint_vec, dataset, active_indices_np, batch_size
        )
        # Map distances back to original indices
        prune_mask_active = dist_from_waypoint_active < dist_from_source[active_indices]
        
        # Update active array using the active indices
        prune_indices = active_indices[prune_mask_active]
        active[prune_indices] = False
    return edges

################################


################################
### HYBRID SET COVER METHODS ###

def greedySetCoverWithFriends(sets, friends, covered):
    n = sets.shape[0]
    edges = []
    
    # Pre-compute friend mask once
    fcover = cp.zeros(n, dtype=cp.bool_)
    fcover[friends] = True
    uncovered = len(friends) - 1
    
    while uncovered > 0:
        # Compute scores using matrix multiplication
        scores = (~covered).astype(cp.int32) @ sets
        
        index = int(cp.argmax(scores))
        edges.append(friends[index])
        
        # Update covered set
        newly_covered = cp.array(sets[:, index], dtype=cp.bool_)
        covered |= newly_covered
        uncovered -= int(cp.sum(newly_covered & fcover))
    
    return edges, covered

def buildHybridSetCover(dataset, k):
    n = dataset.shape[0]
    edgeSet = []
    
    # Pre-allocate distance array
    distanceFromSource = np.empty(n, dtype=np.float32)
    
    for source in tqdm(range(n)):
        covered = cp.zeros(n, dtype=cp.bool_)
        covered[source] = True  # Mark source as covered upfront
        
        # Compute all distances once
        distanceFromSource[:] = cdist(dataset, dataset[source:source+1], metric='euclidean').flatten()
        sourceEdges = []
        
        while int(cp.sum(covered)) < n:
            # Mask covered points
            masked_dist = distanceFromSource.copy()
            masked_dist[covered.get()] = np.inf
            
            # Get k nearest uncovered neighbors (excluding source)
            masked_dist[source] = np.inf
            friends = np.argpartition(masked_dist, min(k, n - int(cp.sum(covered)) - 1))[:k]
            friends = friends[masked_dist[friends] < np.inf]
            
            if len(friends) == 0:
                break
            
            # Always include source in friends
            friends = np.concatenate([[source], friends])
            
            # Compute permutation matrix only for friends
            friend_data = dataset[friends]
            distances = cdist(dataset, friend_data, metric='euclidean')
            permutation_matrix = cp.asarray(rankdata(distances, method='ordinal', axis=1), dtype=cp.uint16)
            source_idx = 0
                
            sets = buildSetsOfSource(permutation_matrix, source_idx)
            edges, covered = greedySetCoverWithFriends(sets, friends, covered)
            sourceEdges.extend(edges)
        
        edgeSet.append(sourceEdges)
    
    return edgeSet

def buildHybridSetCoverWithBetterFriends(dataset, k, v):
    n = dataset.shape[0]
    edgeSet = []

    for source in tqdm(range(n)):
        covered = cp.zeros(n, dtype=cp.bool_)
        covered[source] = True

        # Compute distances once per source
        distanceFromSource = cdist(dataset, dataset[source:source+1], metric='euclidean').ravel()
        sourceEdges = []

        while int(cp.sum(covered)) < n:
            uncovered_mask = ~covered.get()
            uncovered_mask[source] = False
            uncovered_indices = np.where(uncovered_mask)[0]

            if len(uncovered_indices) == 0:
                break

            # Vote efficiently
            voters = np.random.choice(uncovered_indices, size=min(v, len(uncovered_indices)), replace=False)
            voter_dists = cdist(dataset, dataset[voters], metric='euclidean')
            votes = cp.sum(voter_dists < distanceFromSource[:, None], axis=1)

            # Get top k friends from uncovered points
            possible_friends = min(k, len(uncovered_indices))
            uncovered_votes = votes[uncovered_mask]
            top_k_local = np.argpartition(uncovered_votes, -possible_friends)[-possible_friends:]
            friends = uncovered_indices[top_k_local]

            friends = np.concatenate([[source], friends])

            # Compute distances only for friends
            distances = cdist(dataset, dataset[friends], metric='euclidean')
            permutation_matrix = cp.asarray(rankdata(distances, method='ordinal', axis=1), dtype=cp.uint16)

            sets = buildSetsOfSource(permutation_matrix, 0)
            edges, covered = greedySetCoverWithFriends(sets, friends, covered)
            sourceEdges.extend(edges)

        edgeSet.append(sourceEdges)

    return edgeSet

### CHUNKING FOR MEMORY EFFICIENT ROBUST PRUNE ###

def auto_chunk_size(dataset_shape, safety_factor=0.6):
    """Automatically determine optimal chunk size"""
    n, d = dataset_shape
    
    # Get available RAM
    available_ram = psutil.virtual_memory().available
    
    # Memory needed per row: 
    # - n * 8 bytes for distances (float64)
    # - n * ceil(log_2(n)/8) bytes for ranks (bits converted to bytes)
    bytes_for_ranks = n * np.ceil(np.log2(n) / 8)
    bytes_per_row = n * 8 + bytes_for_ranks
    
    # Calculate chunk size
    max_chunk = int((available_ram * safety_factor) / bytes_per_row)
    
    # Apply practical limits
    chunk_size = max(100, min(max_chunk, n, 10000))
    
    print(f"Dataset: {n} samples, {d} features")
    print(f"Available RAM: {available_ram / 1e9:.1f} GB")
    print(f"Bits per rank: {np.log2(n):.2f}")
    print(f"Bytes per rank: {np.ceil(np.log2(n) / 8):.0f}")
    print(f"Recommended chunk_size: {chunk_size}")
    
    return chunk_size

def checkNavigability(dist, graph):
    n = dist.shape[0]
    for i in tqdm(range(n)):
        # distances = np.array([np.linalg.norm(dataset[i] - dataset[e[1]]) for e in graph[i])
        for j in np.random.choice(n, 1000, replace=False):
            if j == i:
                continue
            distances = np.array([dist[j][e] for e in graph[i]])
            if not np.any(distances <= dist[i][j]):
                print("Found no greedy edge. G is not navigable.")
                print(i, j, dist[i][j], '\n', distances)
                return

    print("G appears to be navigable")
