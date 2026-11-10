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

Pieces (local_dim_pipeline.py runs them; effective_dim.ipynb plots what it writes):
  computed_out_degrees  sources and out-degrees: line i of the adjacency list is the out-neighbourhood of
                        the point on line i of a separate computed file
  NpyFile, iter_blocks  rows of vectors.npy read with plain file I/O, the next block read in a background
                        thread while the device works on the current one
  knn_vectors           one sequential pass: the k nearest neighbours of every query, keeping their vectors
                        (not their ids) on the device, so nothing is read back from the file afterwards
  local_pr_batch        PR for every k (prefixes of the K nearest) and both centrings
  host_available_bytes, device_free_bytes   memory budgets for sizing blocks

Run `python src/localdim.py` for the sanity checks.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CENTRES = ("mean", "v")


def resolve_device(device=None):
    """None / "auto" -> "cuda" if torch sees a GPU, else "cpu"; anything else is returned unchanged."""
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


### Logging ###

def log(msg):
    """Print msg with a timestamp, flushed (shows up at once in a SLURM log)."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def track(iterable, total, desc, unit="it", mininterval=30):
    """tqdm when stderr is a terminal (or in a notebook); otherwise, e.g. in a SLURM log, one timestamped
    line every `mininterval` seconds and one at the end: count / total (percent), elapsed, time left, rate.
    Plain lines instead of tqdm's carriage-return redraws, so `tail -f` and grep work on the log."""
    interactive = sys.stderr.isatty() or "ipykernel" in sys.modules
    if interactive:
        from tqdm.auto import tqdm
        yield from tqdm(iterable, total=total, desc=desc, unit=unit, mininterval=min(mininterval, 1))
        return
    t0 = last = time.time()
    log(f"{desc}: 0/{total:,} {unit}s")
    i = 0
    for i, item in enumerate(iterable, 1):
        yield item
        now = time.time()
        if now - last >= mininterval or i == total:
            last, el = now, now - t0
            rate = i / el if el > 0 else float("inf")
            left = f", ~{_hms((total - i) / rate)} left" if i < total and rate > 0 else ""
            log(f"{desc}: {i:,}/{total:,} {unit}s ({100 * i / total:.0f}%), {_hms(el)} elapsed{left}, "
                f"{rate:,.1f} {unit}/s")


### Memory ###

def _read_int(path):
    try:
        s = Path(path).read_text().strip()
        return None if s in ("", "max") else int(s)
    except (OSError, ValueError):
        return None


def _cgroup_available():
    """Bytes left under this process's cgroup memory limit (e.g. a SLURM job), or None if unlimited/unknown.
    Reclaimable page cache (inactive file pages) is not counted as used: reading vectors.npy fills it."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:                                   # cgroup v2: "0::/path"
        if line.startswith("0::"):
            base = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            limit, used = _read_int(base / "memory.max"), _read_int(base / "memory.current")
            stat = base / "memory.stat"
            if limit is not None and used is not None:
                inactive = 0
                try:
                    for s in stat.read_text().splitlines():
                        if s.startswith("inactive_file "):
                            inactive = int(s.split()[1])
                except OSError:
                    pass
                return max(0, limit - used + inactive)
    for line in lines:                                   # cgroup v1: "N:memory:/path"
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            base = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
            limit = _read_int(base / "memory.limit_in_bytes")
            used = _read_int(base / "memory.usage_in_bytes")
            if limit is not None and used is not None and limit < 1 << 60:
                return max(0, limit - used)
    return None


def host_available_bytes():
    """RAM this process can still use: the smallest of the system's available memory (MemAvailable, or
    psutil, or physical size / 2 as a last resort), the cgroup limit headroom and SLURM_MEM_PER_NODE."""
    cands = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                cands.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    if not cands:
        try:
            import psutil
            cands.append(psutil.virtual_memory().available)
        except ImportError:
            try:
                cands.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2)
            except (ValueError, OSError, AttributeError):
                cands.append(8 << 30)
    cg = _cgroup_available()
    if cg is not None:
        cands.append(cg)
    if os.environ.get("SLURM_MEM_PER_NODE", "").isdigit():
        cands.append(int(os.environ["SLURM_MEM_PER_NODE"]) << 20)       # MB
    return min(cands)


def device_free_bytes(device):
    """Free memory on a CUDA device (after releasing torch's cache), or host_available_bytes() otherwise."""
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info(torch.device(device))[0]
    return host_available_bytes()


### Graph ###

def parse_neighbours(line):
    """Neighbour ids on one adjacency-list line. Accepts "[a, b, ...]", "[(a, u), (b, u), ...]" (the first
    of each pair is the neighbour, as in vibe.py) and bare "a b ..." / "a, b, ..."; anything before the
    first "[" is ignored."""
    body = line.strip()
    if "[" in body:
        body = body[body.index("[") + 1 : body.rindex("]")]
    if "(" in body:
        pairs = np.array(body.replace("(", " ").replace(")", " ").replace(",", " ").split(), dtype=np.int64)
        return pairs[0::2]
    return np.array(body.replace(",", " ").split(), dtype=np.int64)


def computed_out_degrees(adj_path, computed_path):
    """(sources, out-degrees), sources sorted, for every point with a computed out-neighbourhood.

    Line i of `adj_path` is the out-neighbourhood of point int(line i of `computed_path`); the two files
    must have the same number of (non-blank) lines. A point listed twice (e.g. a resumed run) keeps its
    last line, with a warning."""
    with open(computed_path) as f:
        ids = [int(s) for s in (l.strip() for l in f) if s]
    with open(adj_path) as f:
        degs = [len(parse_neighbours(l)) for l in f if l.strip()]
    if len(ids) != len(degs):
        raise ValueError(f"{computed_path} has {len(ids):,} lines but {adj_path} has {len(degs):,}; "
                         "line i of one must match line i of the other")
    deg_of = dict(zip(ids, degs))                         # later lines win
    if len(deg_of) < len(ids):
        print(f"warning: {len(ids) - len(deg_of):,} points appear more than once in {computed_path}; "
              "kept the last line of each")
    sources = np.array(sorted(deg_of), dtype=np.int64)
    out_deg = np.array([deg_of[s] for s in sources], dtype=np.int64)
    return sources, out_deg


### Reading vectors.npy ###

class NpyFile:
    """An (n, d) C-order .npy file: shape, dtype, and contiguous row ranges read with plain file I/O
    (`read` releases the GIL, so a background thread can read ahead). `rows(ids)` reads a few scattered rows
    (sorted ids) through a memory map."""

    def __init__(self, path):
        self.path = Path(path)
        m = np.load(self.path, mmap_mode="r")
        if m.ndim != 2 or not m.flags.c_contiguous:
            raise ValueError(f"{path}: expected a C-order (n, d) array, got shape {m.shape}")
        self.shape, self.dtype, self.offset = m.shape, m.dtype, m.offset
        self.row_bytes = m.shape[1] * m.dtype.itemsize
        self._mmap = m

    def __len__(self):
        return self.shape[0]

    def read(self, start, stop, out):
        """Rows start .. stop - 1 into the first stop - start rows of `out`; returns that view."""
        view = out[: stop - start]
        buf = memoryview(view.reshape(-1).view(np.uint8))
        with open(self.path, "rb", buffering=0) as f:
            f.seek(self.offset + start * self.row_bytes)
            got = 0
            while got < len(buf):
                r = f.readinto(buf[got:])
                if not r:
                    raise EOFError(f"{self.path}: short read at row {start}")
                got += r
        return view

    def rows(self, ids):
        return np.ascontiguousarray(self._mmap[np.asarray(ids, dtype=np.int64)])


def iter_blocks(f, stop, chunk, start=0):
    """(start, stop, rows) for consecutive blocks of `chunk` rows of NpyFile f, from `start` to `stop`.
    Double buffered: block i + 1 is read in a thread while the caller uses block i, and the caller must be
    done with a block (e.g. copied it to the device) before asking for the next one."""
    bufs = [np.empty((min(chunk, stop - start), f.shape[1]), dtype=f.dtype) for _ in range(2)]
    starts = list(range(start, stop, chunk))
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(f.read, starts[0], min(starts[0] + chunk, stop), bufs[0])
        for i, s in enumerate(starts):
            rows = fut.result()
            if i + 1 < len(starts):
                nxt = starts[i + 1]
                fut = ex.submit(f.read, nxt, min(nxt + chunk, stop), bufs[(i + 1) % 2])
            yield s, min(s + chunk, stop), rows


### Nearest neighbours, keeping their vectors ###

def knn_vectors(f, rows, q_ids, Q_raw, k, device, chunk, query_batch, progress=True, mininterval=0.1):
    """The k nearest neighbours (squared Euclidean) of each query among rows 0 .. rows - 1 of NpyFile f, a
    query itself excluded: their vectors (in the file's dtype) and squared distances, nearest first.

    q_ids: sorted row ids of the queries; Q_raw: their vectors (len(q_ids), d). One sequential pass over the
    file in blocks of `chunk` rows. For every query the running top k (distances and vectors) stays on
    `device`, about k * (4 + d * itemsize) bytes per query. Per block and query batch, only queries with a
    point closer than their current k-th neighbour are merged, so late blocks cost little beyond the
    distance computation. Distances are float32. Returns torch tensors (vectors (q, k, d), dists (q, k))."""
    import torch
    nq, d = Q_raw.shape
    assert np.all(np.diff(q_ids) > 0), "queries must be sorted and unique"
    assert k <= rows - 1, f"k = {k} needs at least {k + 1} points, have {rows:,}"
    Qr = torch.as_tensor(np.ascontiguousarray(Q_raw), device=device)
    Q = Qr.float()
    q_sq = (Q * Q).sum(1)
    own_all = torch.as_tensor(np.asarray(q_ids, dtype=np.int64), device=device)
    best_d = torch.full((nq, k), float("inf"), device=device)
    best_v = torch.zeros((nq, k, d), dtype=Qr.dtype, device=device)

    blocks = iter_blocks(f, rows, chunk)
    if progress:
        blocks = track(blocks, -(-rows // chunk), f"knn {nq:,} queries, k={k}", "block", mininterval)
    for s, e, block in blocks:
        Xr = torch.from_numpy(block).to(device)           # synchronous: the buffer can be refilled after this
        X = Xr.float()
        x_sq = (X * X).sum(1)
        kk = min(k, e - s)
        for qs in range(0, nq, query_batch):
            qe = min(qs + query_batch, nq)
            D = Q[qs:qe] @ X.T
            D.mul_(-2).add_(q_sq[qs:qe, None]).add_(x_sq[None])
            own = own_all[qs:qe]
            r = torch.nonzero((own >= s) & (own < e)).squeeze(1)
            D[r, own[r] - s] = float("inf")                 # a query is not its own neighbour
            d_blk, i_blk = D.topk(kk, dim=1, largest=False)
            del D
            cand = torch.nonzero(d_blk[:, 0] < best_d[qs:qe, -1]).squeeze(1)
            if len(cand) == 0:
                continue
            rws = qs + cand
            d_all = torch.cat([best_d[rws], d_blk[cand]], 1)
            d_new, pos = d_all.topk(k, dim=1, largest=False)  # sorted, nearest first
            from_blk = pos >= k
            v = best_v[rws[:, None], torch.where(from_blk, 0, pos)]    # kept entries (others overwritten next)
            rr, cc = torch.nonzero(from_blk, as_tuple=True)
            v[rr, cc] = Xr[i_blk[cand][rr, pos[rr, cc] - k]]
            best_d[rws], best_v[rws] = d_new, v
            del d_all, d_new, pos, from_blk, v
        del Xr, X, x_sq
    return best_v, best_d


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


def local_pr_batch(Y, v, ks, centres=CENTRES, estimator="plugin", dtype="float64"):
    """{(centre, k): PR per source (numpy)} from neighbour vectors Y (b, >= max(ks), d), nearest first, and
    source vectors v (b, d), both torch tensors on the same device; every k is a prefix of Y."""
    import torch
    dt = getattr(torch, dtype)
    Y, v = Y.to(dt), v.to(dt)
    out = {}
    for k in ks:
        Yk = Y[:, :k]
        for c in centres:
            ref = Yk.mean(1, keepdim=True) if c == "mean" else v[:, None]
            out[(c, k)] = pr_rows(Yk - ref, estimator).cpu().numpy()
    return out


def k_grid(base, d, d_relative=(0.5, 1, 2), n=None):
    """Sorted union of `base` and round(f * d) for f in d_relative, keeping k <= n - 1 when n is given."""
    ks = set(int(k) for k in base) | {max(2, round(f * d)) for f in d_relative}
    return sorted(k for k in ks if n is None or k <= n - 1)


### Sanity checks ###

def _check():
    import tempfile

    import torch
    from effdim import participation_ratio
    with tempfile.TemporaryDirectory() as tmp:
        adj, comp = Path(tmp) / "adj-list-x.txt", Path(tmp) / "x-computed.txt"
        adj.write_text("[3, 4, 5]\n[(1, 9), (2, 0)]\n7 8 9 10\n[]\n[1]\n")
        comp.write_text("12\n2\n40\n5\n2\n")              # point 2 twice: its last line ([1]) wins
        s, deg = computed_out_degrees(adj, comp)
        assert s.tolist() == [2, 5, 12, 40] and deg.tolist() == [1, 0, 3, 4], (s, deg)
        print("computed_out_degrees: line-aligned files read correctly")

        rng = np.random.default_rng(0)
        n, d, K = 3000, 24, 40
        for dtype in (np.float32, np.uint8):
            V = (rng.standard_normal((n, d)) * np.linspace(3, 0.2, d) * 20 + 128).clip(0, 255).astype(dtype)
            np.save(Path(tmp) / "v.npy", V)
            f = NpyFile(Path(tmp) / "v.npy")
            assert np.array_equal(np.concatenate([b.copy() for _, _, b in iter_blocks(f, n, 700)]), V)
            queries = np.sort(rng.choice(n, 50, replace=False))
            Y, dist = knn_vectors(f, n, queries, f.rows(queries), K, "cpu", chunk=700, query_batch=17, progress=False)
            Dm = ((V[queries, None, :].astype(np.float64) - V[None].astype(np.float64)) ** 2).sum(2)
            Dm[np.arange(len(queries)), queries] = np.inf
            ref = np.sort(Dm, 1)[:, :K]
            assert np.allclose(dist.numpy(), ref, rtol=1e-4, atol=1e-2), "knn distances disagree with brute force"
            Yn = Y.numpy().astype(np.float64)
            back = ((Yn - V[queries, None, :].astype(np.float64)) ** 2).sum(2)
            assert np.allclose(back, dist.numpy(), rtol=1e-4, atol=1e-2), "kept vectors don't match their distances"
            assert Y.dtype == torch.from_numpy(V[:1]).dtype
        print("knn_vectors: distances match brute force, kept vectors match their distances (float32, uint8)")

        V = V.astype(np.float32)
        ks = [10, 20, 40]                               # k < d, k ~ d, k > d: both Gram branches
        Yt, vt = Y.float(), torch.from_numpy(V[queries])
        pr = local_pr_batch(Yt, vt, ks, estimator="plugin")
        pu = local_pr_batch(Yt, vt, ks, estimator="u")
        for i in range(0, len(queries), 13):
            for k in ks:
                Yk = Yt[i, :k].numpy().astype(np.float64)
                for c, Z in (("mean", Yk - Yk.mean(0)), ("v", Yk - V[queries[i]])):
                    lam = np.linalg.eigvalsh(Z.T @ Z)
                    assert np.isclose(pr[(c, k)][i], lam.sum() ** 2 / (lam**2).sum()), (c, k)
                    assert pr[(c, k)][i] <= pr_cap(k, d, c) + 1e-9
                assert np.isclose(pu[("mean", k)][i], participation_ratio(Yk)), k
        print("local_pr_batch: plug-in matches eigenvalues, 'u' matches effdim.participation_ratio, caps hold")
    print(f"host_available_bytes: {host_available_bytes() / 2**30:.1f} GiB")


if __name__ == "__main__":
    _check()
