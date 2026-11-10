"""
Beam search on a graph over single-vector embeddings, with a trace of everything the search touched,
so we can ask how often a relevant document is seen, enters the beam, is expanded, or ends up in the
results. Used by obliq_beam_search.ipynb.

collapse()     multi-vector item -> one vector: mean of the item's token vectors, optionally L2-normalised
medoid()       the point closest to the centroid (the usual fixed entry point)
exact_ranks()  brute-force squared-Euclidean distances and every item's rank, per query
beam_search()  greedy beam search (beam width L, as in DiskANN / Vamana) under squared Euclidean distance
"""

import numpy as np


def collapse(vectors, offsets, normalize=True, chunk=1 << 20):
    """Mean of each item's vectors (rows offsets[i]:offsets[i+1]) as float32 [n_items, dim], L2-normalised
    if normalize. `vectors` may be a memmap; it is read in blocks of about `chunk` rows."""
    counts = np.diff(offsets)
    if (counts <= 0).any():
        raise ValueError("every item needs at least one vector")
    n = len(counts)
    out = np.empty((n, vectors.shape[1]), dtype=np.float32)
    i = 0
    while i < n:
        j = min(max(int(np.searchsorted(offsets, offsets[i] + chunk, side="right")) - 1, i + 1), n)
        block = np.asarray(vectors[offsets[i]:offsets[j]], dtype=np.float32)
        out[i:j] = np.add.reduceat(block, offsets[i:j] - offsets[i], axis=0)
        i = j
    out /= counts[:, None]
    if normalize:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def medoid(X):
    """Index of the point closest to the mean of X."""
    return int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))


def sq_dists(Q, X):
    """Squared Euclidean distances [len(Q), len(X)], clipped at 0."""
    D = (Q ** 2).sum(1)[:, None] - 2 * Q @ X.T + (X ** 2).sum(1)[None, :]
    return np.maximum(D, 0, out=D)


def exact_ranks(Q, X, excluded=None):
    """(D, ranks): squared distances [len(Q), len(X)] and each item's 0-based rank per query. excluded:
    one array of item indices per query (or None); those items get distance inf and rank -1 and do not
    take up ranks, as they are removed from results at search time."""
    D = sq_dists(Q, X)
    if excluded is not None:
        for qi, ex in enumerate(excluded):
            D[qi, ex] = np.inf
    order = np.argsort(D, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(X.shape[0])[None, :], axis=1)
    ranks[~np.isfinite(D)] = -1
    return D, ranks


def beam_search(graph, X, q, L, entry):
    """Greedy beam search for q from `entry` with beam width L: repeatedly expand the closest unexpanded
    beam node, compute distances to its not-yet-seen out-neighbours, keep the L closest of everything in
    the beam, and stop when every beam node is expanded. graph: one out-neighbour index array per node.

    Returns (beam_ids, beam_dists, trace). The beam is sorted by distance. trace holds int arrays over all
    n nodes, -1 meaning never:
      seen      expansion step at which the node's distance was computed (0 = the entry point)
      inserted  step at which it entered the beam (a node is scored once, so this is its seen step or -1)
      expanded  step at which it was expanded
      touches   how many expanded nodes list it as an out-neighbour (counts repeat encounters)
    and the totals steps (expansions) and dists (distance computations)."""
    n = len(X)
    seen = np.full(n, -1, dtype=np.int64)
    inserted = np.full(n, -1, dtype=np.int64)
    expanded = np.full(n, -1, dtype=np.int64)
    touches = np.zeros(n, dtype=np.int64)

    seen[entry] = inserted[entry] = 0
    beam_ids = np.array([entry], dtype=np.int64)
    beam_d = np.array([((X[entry] - q) ** 2).sum()], dtype=np.float32)
    done = np.zeros(1, dtype=np.bool_)
    step, dists = 0, 1
    while not done.all():
        b = int(np.argmin(done))                  # beam is sorted: first unexpanded node is the closest
        u = beam_ids[b]
        done[b] = True
        step += 1
        expanded[u] = step
        nbrs = graph[u]
        touches[nbrs] += 1
        new = nbrs[seen[nbrs] < 0]
        if not len(new):
            continue
        seen[new] = step
        dists += len(new)
        ids = np.concatenate([beam_ids, new])
        d = np.concatenate([beam_d, ((X[new] - q) ** 2).sum(1)])
        keep = np.argsort(d, kind="stable")[:L]
        beam_ids, beam_d = ids[keep], d[keep]
        done = np.concatenate([done, np.zeros(len(new), dtype=np.bool_)])[keep]
        entered = beam_ids[inserted[beam_ids] < 0]
        inserted[entered] = step

    trace = {"seen": seen, "inserted": inserted, "expanded": expanded, "touches": touches,
             "steps": step, "dists": dists}
    return beam_ids, beam_d, trace
