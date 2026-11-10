"""
Local effective dimension: the participation ratio (PR) of each point's k nearest neighbours, for the
points whose out-neighbourhood has been computed in a graph, to compare with their out-degree.

For a source v with nearest neighbours x_1..x_k (exact, squared Euclidean, v itself excluded), the k x d
block Z is centred in one of two ways (CENTRES):
  "mean"  Z = x_j - mean_j x_j   local PCA: how many directions the neighbourhood spreads along
  "v"     Z = x_j - v            directions out of v: what robust prune at v works with
PR = (tr C)^2 / tr(C^2) with C = Z^T Z (the 1/k scale cancels). It is computed from the Gram matrix,
Z Z^T (k x k) or Z^T Z (d x d), whichever is smaller: tr C = tr G and tr C^2 = ||G||_F^2.

estimator="plugin" is the PR of the k points themselves, so it is at most rank(Z): min(k - 1, d) for
"mean", min(k, d) for "v". Read small-k values against that cap (`pr_cap`). estimator="u" leaves out
the i = j terms (the U-statistic of effdim.participation_ratio, identical for "mean"), which estimates
the PR of the local distribution instead; it is not capped but is noisy when k << d.

Steps: `computed_out_degrees` (sources and out-degrees from an adjacency list in vibe.read_graph format,
restricted to the computed-sources file), `knn` (one pass over memory-mapped vectors in row blocks,
running top-k for all sources on a torch device), `local_pr` (all k and both centrings from one gather
of the K = max(k) nearest neighbours per source).

Used by effective_dim.ipynb. Run `python src/localdim.py` for the sanity checks.
"""

from pathlib import Path

import numpy as np

CENTRES = ("mean", "v")


def resolve_device(device=None):
    """None / "auto" -> "cuda" if torch sees a GPU, else "cpu"; anything else is returned unchanged."""
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


### Graph ###

def computed_path_for(adj_path):
    """The computed-sources file that goes with adj-list-<stem>.txt (vibe.graph_paths naming), or None."""
    adj_path = Path(adj_path)
    if not adj_path.name.startswith("adj-list-"):
        return None
    return adj_path.with_name(adj_path.name.removeprefix("adj-list-").removesuffix(".txt") + "-computed.txt")


def computed_out_degrees(adj_path, computed_path="auto"):
    """(sources, out-degrees), sources sorted, for every source with a computed out-neighbourhood.

    Reads the adjacency list with vibe.read_graph. If a computed-sources file exists ("auto": the one named
    like the adjacency list, see computed_path_for; None: don't use one), only sources listed in it are
    kept: a run interrupted between writing a batch's adjacency lines and its computed list leaves sources
    in the first but not the second."""
    from vibe import read_graph
    hoods = read_graph(adj_path)
    if computed_path == "auto":
        computed_path = computed_path_for(adj_path)
    if computed_path is not None and Path(computed_path).exists():
        done = set(map(int, Path(computed_path).read_text().split()))
        dropped = len(hoods) - sum(s in done for s in hoods)
        if dropped:
            print(f"{adj_path}: {dropped:,} sources are in the adjacency list but not in {computed_path}; dropped")
        hoods = {s: h for s, h in hoods.items() if s in done}
    sources = np.array(sorted(hoods), dtype=np.int64)
    out_deg = np.array([len(hoods[s]) for s in sources], dtype=np.int64)
    return sources, out_deg


### Nearest neighbours ###

def knn(V, queries, k, rows=None, device=None, chunk=1 << 17, query_batch=2048, progress=True):
    """Exact k nearest neighbours (squared Euclidean) of the rows `queries` (sorted row indices) of V among
    rows 0 .. rows - 1 of V (default: all of them), each query itself excluded.

    One pass over V in blocks of `chunk` rows, so a memory-mapped billion-row file is read once, with a
    running top-k for every query kept on `device`. Distances are float32 (|x|^2 - 2<x, q> + |q|^2).
    Device memory is about query_batch * chunk * 4 bytes * 3 for one block, plus len(queries) * k * 12.
    Returns (ids (q, k) int64, squared distances (q, k) float32), nearest first."""
    import torch
    device = resolve_device(device)
    n = len(V) if rows is None else int(rows)
    queries = np.asarray(queries, dtype=np.int64)
    assert np.all(np.diff(queries) > 0), "queries must be sorted and unique"
    assert queries[-1] < n, f"query row {queries[-1]} is outside the first {n:,} rows"
    assert k <= n - 1, f"k = {k} needs at least k + 1 = {k + 1} points, have {n:,}"

    q_ids = torch.as_tensor(queries, device=device)
    Q = torch.as_tensor(np.asarray(V[queries]), device=device).float()
    q_sq = (Q * Q).sum(1)
    best_d = torch.full((len(Q), k), float("inf"), device=device)
    best_i = torch.full((len(Q), k), -1, dtype=torch.int64, device=device)

    starts = range(0, n, chunk)
    if progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, unit="block", desc=f"knn k={k}")
    for start in starts:
        stop = min(start + chunk, n)
        X = torch.as_tensor(np.ascontiguousarray(V[start:stop]), device=device).float()
        x_sq = (X * X).sum(1)
        kk = min(k, stop - start)
        for qs in range(0, len(Q), query_batch):
            qe = min(qs + query_batch, len(Q))
            D = q_sq[qs:qe, None] - 2 * (Q[qs:qe] @ X.T) + x_sq[None]
            own = q_ids[qs:qe]
            r = torch.nonzero((own >= start) & (own < stop)).squeeze(1)
            D[r, own[r] - start] = float("inf")                     # a query is not its own neighbour
            d_blk, i_blk = D.topk(kk, dim=1, largest=False)
            d_all = torch.cat([best_d[qs:qe], d_blk], 1)
            i_all = torch.cat([best_i[qs:qe], i_blk + start], 1)
            d_new, pos = d_all.topk(k, dim=1, largest=False)          # sorted, nearest first
            best_d[qs:qe], best_i[qs:qe] = d_new, i_all.gather(1, pos)
            del D, d_blk, i_blk, d_all, i_all
    return best_i.cpu().numpy(), best_d.cpu().numpy()


### Local PR ###

def pr_rows(Z, estimator="plugin"):
    """PR of each (k, d) block of a (b, k, d) torch tensor whose rows are already centred."""
    k, d = Z.shape[1:]
    G = Z @ Z.mT if k <= d else Z.mT @ Z
    tr = G.diagonal(dim1=1, dim2=2).sum(1)
    fro2 = (G * G).sum((1, 2))
    if estimator == "plugin":
        return tr**2 / fro2
    if estimator == "u":                                            # drop the i = j terms
        sq = (Z * Z).sum(2)
        s2 = (sq * sq).sum(1)
        return (tr**2 - s2) / (fro2 - s2)
    raise ValueError(f"estimator must be 'plugin' or 'u', got {estimator!r}")


def pr_cap(k, d, centre):
    """Largest possible plug-in PR of k points in d dimensions (the rank of Z)."""
    return min(k - 1 if centre == "mean" else k, d)


def local_pr(V, sources, nbrs, ks, centres=CENTRES, estimator="plugin", device=None, batch=64,
             dtype="float64", progress=True):
    """{(centre, k): PR per source} for every k in ks and centre in centres, with nbrs (q, >= max(ks))
    the sources' nearest neighbours, nearest first (from knn). Each batch of sources gathers its K = max(ks)
    neighbours from V once (sorted reads, memmap friendly) and takes every k as a prefix. Device memory is
    about batch * K * d * 8 bytes (float64) plus the batch's Gram matrices."""
    import torch
    device = resolve_device(device)
    dt = getattr(torch, dtype)
    ks = sorted(set(int(k) for k in ks))
    K = ks[-1]
    assert nbrs.shape[1] >= K, f"have {nbrs.shape[1]} neighbours per source, need {K}"
    sources = np.asarray(sources, dtype=np.int64)
    assert np.all(np.diff(sources) > 0), "sources must be sorted and unique (as knn returns them)"
    out = {(c, k): np.empty(len(sources)) for c in centres for k in ks}

    starts = range(0, len(sources), batch)
    if progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, unit="batch", desc="local PR")
    for s in starts:
        e = min(s + batch, len(sources))
        ids = nbrs[s:e, :K]
        u, inv = np.unique(ids, return_inverse=True)
        U = torch.as_tensor(np.asarray(V[u]), device=device).to(dt)
        Y = U[torch.as_tensor(inv.reshape(ids.shape), device=device)]                     # (b, K, d)
        v = torch.as_tensor(np.asarray(V[sources[s:e]]), device=device).to(dt)
        for k in ks:
            Yk = Y[:, :k]
            for c in centres:
                ref = Yk.mean(1, keepdim=True) if c == "mean" else v[:, None]
                out[(c, k)][s:e] = pr_rows(Yk - ref, estimator).cpu().numpy()
        del U, Y, v
    return out


def k_grid(base, d, d_relative=(0.5, 1, 2), n=None):
    """Sorted union of `base` and round(f * d) for f in d_relative, keeping k <= n - 1 when n is given."""
    ks = set(int(k) for k in base) | {max(2, round(f * d)) for f in d_relative}
    return sorted(k for k in ks if n is None or k <= n - 1)


### Sanity checks ###

def _check():
    from effdim import participation_ratio
    rng = np.random.default_rng(0)
    n, d = 3000, 24
    V = (rng.standard_normal((n, d)) * np.linspace(3, 0.2, d)).astype(np.float32)
    queries = np.sort(rng.choice(n, 50, replace=False))
    ids, dist = knn(V, queries, 40, device="cpu", chunk=700, query_batch=17, progress=False)
    D = ((V[queries, None, :].astype(np.float64) - V[None]) ** 2).sum(2)
    D[np.arange(len(queries)), queries] = np.inf
    ref = np.argsort(D, 1, kind="stable")[:, :40]
    assert (np.sort(ids, 1) == np.sort(ref, 1)).mean() > 0.99, "knn disagrees with brute force"
    assert not np.isin(queries[:, None], ids).any(axis=1).any(), "a query is its own neighbour"
    print("knn: matches brute force")

    ks = [10, 20, 40]                                   # k < d, k ~ d, k > d: both Gram branches
    pr = local_pr(V, queries, ids, ks, estimator="plugin", device="cpu", batch=7, progress=False)
    pu = local_pr(V, queries, ids, ks, estimator="u", device="cpu", batch=7, progress=False)
    for i in range(0, len(queries), 13):
        for k in ks:
            Y = V[ids[i, :k]].astype(np.float64)
            for c, Z in (("mean", Y - Y.mean(0)), ("v", Y - V[queries[i]])):
                lam = np.linalg.eigvalsh(Z.T @ Z)
                assert np.isclose(pr[(c, k)][i], lam.sum() ** 2 / (lam**2).sum()), (c, k)
                assert pr[(c, k)][i] <= pr_cap(k, d, c) + 1e-9
            assert np.isclose(pu[("mean", k)][i], participation_ratio(Y)), k
    print("local_pr: plug-in matches eigenvalues, 'u' matches effdim.participation_ratio, caps hold")


if __name__ == "__main__":
    _check()
