"""
VIBE dataset vs a Gaussian with the same covariance spectrum: robust-prune graphs and their statistics.

Shared by vibe_robust_prune.ipynb (interactive, with plots) and vibe_pipeline.py (batch / SLURM). Both
write the same files to graph_dir, so either one picks up what the other has already computed:

  <name>-meta.json                             n, d, distance and the run's settings
  <name>-spectrum.npy                          dataset covariance eigenvalues; the Gaussian is built from these
  <name>-gaussian-spectrum.npy                 the Gaussian's own sample covariance eigenvalues
  adj-list-<name>-alpha1-euclidean.txt         one line per source: "source [(neighbor, uncovered), ...]", where
                                               uncovered = points still uncovered once that edge is added
                                               (the format of simulrun.py); the last one is 0
  <name>-alpha1-euclidean-computed.txt         sources already done (an interrupted run resumes)
  adj-list-<name>-gaussian-alpha1-...          the same two files for the synthetic Gaussian
  <name>-results.csv                           spectrum and graph statistics, one row per dataset

<name> is the VIBE dataset name, plus "-n<k>-seed<s>" when only a random subset of k points is used.

Steps: `download` (standard library only: resumes a .part file, checks size and SHA-256), `load_train`
(L2-normalises cosine / angular datasets so the Euclidean prune gives the cosine graph), `dataset_spectrum`
and `gaussian_like` (same n, same covariance eigenvalues), `load_prune` + `build_graph`
(batchedEuclideanRobustPrune from utils.py on GPU via CuPy, else its NumPy variant), `summarize`.
"""

import hashlib
import json
import sys
import types
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from effdim import accumulate, row_blocks, spectrum, spectrum_summary

REPO_ID = "vector-index-bench/vibe"
EUCLIDEAN_DISTANCES = {"euclidean", "cosine", "angular", "any", "normalized"}


### Download ###

def remote_file_info(filename):
    """(size in bytes, sha256 or None) of a file in the Hugging Face dataset repo."""
    api = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
    req = urllib.request.Request(api, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=60) as r:
        files = {f["path"]: f for f in json.load(r) if f.get("type") == "file"}
    if filename not in files:
        names = sorted(p.removesuffix(".hdf5") for p in files if p.endswith(".hdf5"))
        raise FileNotFoundError(f"{filename} is not in {REPO_ID}. Datasets: {', '.join(names)}")
    f = files[filename]
    return f["size"], (f.get("lfs") or {}).get("oid")


def sha256(path, chunk=1 << 24, mininterval=0.1):
    h = hashlib.sha256()
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc="sha256",
                                     mininterval=mininterval) as bar:
        while buf := f.read(chunk):
            h.update(buf)
            bar.update(len(buf))
    return h.hexdigest()


def download(dataset, dest_dir, verify=True, chunk=1 << 24, mininterval=0.1):
    """Download <dataset>.hdf5 into dest_dir (resuming a partial .part file); returns its path.
    A complete file already in dest_dir is reused without downloading or re-checking."""
    filename, dest_dir = f"{dataset}.hdf5", Path(dest_dir).expanduser()
    dest = dest_dir / filename
    size, digest = remote_file_info(filename)
    if dest.exists() and dest.stat().st_size == size:
        print(f"already downloaded: {dest}")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(filename + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:                                   # stale partial from a different file version
        part.unlink()
        have = 0
    if have < size:
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{filename}"
        headers = {"User-Agent": "python-urllib"} | ({"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            if have and r.status != 206:              # server ignored the range: start over
                have = 0
            with open(part, "ab" if have else "wb") as f, \
                    tqdm(total=size, initial=have, unit="B", unit_scale=True, desc=filename,
                         mininterval=mininterval) as bar:
                while buf := r.read(chunk):
                    f.write(buf)
                    bar.update(len(buf))
    if part.stat().st_size != size:
        raise IOError(f"{part}: {part.stat().st_size:,} of {size:,} bytes; run again to resume")
    if verify and digest and sha256(part, mininterval=mininterval) != digest:
        part.unlink()
        raise IOError(f"{filename}: SHA-256 mismatch (partial file deleted); run again to download again")
    part.rename(dest)
    return dest


def describe(path):
    """(attributes, {name: (shape, dtype)}) of a VIBE HDF5 file."""
    with h5py.File(path, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        contents = {k: (f[k].shape, str(f[k].dtype)) for k in f.keys()}
    return attrs, contents


### Data, spectrum, synthetic Gaussian ###

def load_train(path, max_points=None, seed=0):
    """`train` as float32 (all of it, or a seeded random subset of max_points rows), L2-normalised for
    cosine / angular datasets. Returns (X, name, attrs)."""
    attrs, _ = describe(path)
    distance = str(attrs.get("distance", "")).lower()
    point_type = str(attrs.get("point_type", "float")).lower()
    dataset = Path(path).name.removesuffix(".hdf5")
    if distance not in EUCLIDEAN_DISTANCES or point_type == "binary":
        raise ValueError(f"{dataset}: distance={distance!r}, point_type={point_type!r}; robust prune here is "
                         "Euclidean (datasets with distance euclidean, cosine, angular or any)")
    with h5py.File(path, "r") as f:
        train = f["train"]
        n_all = train.shape[0]
        if max_points is not None and max_points < n_all:
            rows = np.sort(np.random.default_rng(seed).choice(n_all, max_points, replace=False))
            X = train[rows]                           # h5py reads sorted index lists
        else:
            X = train[...]
    X = np.ascontiguousarray(X, dtype=np.float32)
    if distance in ("cosine", "angular"):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.where(norms > 0, norms, 1)
    name = dataset if len(X) == n_all else f"{dataset}-n{len(X)}-seed{seed}"
    return X, name, attrs


def dataset_spectrum(X, name, graph_dir):
    """Exact covariance eigenvalues (largest first) and d_eff of X. The eigenvalues are saved to
    <name>-spectrum.npy; if that file already exists, its values are returned instead, so every run
    (notebook or script, on any machine) builds the identical Gaussian."""
    acc = accumulate(row_blocks(X), len(X), name, device="numpy")
    lam = spectrum(acc.covariance())
    saved = Path(graph_dir) / f"{name}-spectrum.npy"
    if saved.exists():
        lam_saved = np.load(saved)
        assert lam_saved.shape == lam.shape, f"{saved} has {lam_saved.shape}, data gives {lam.shape}"
        lam = lam_saved
    else:
        saved.parent.mkdir(parents=True, exist_ok=True)
        np.save(saved, lam)
    return lam, acc.value()


def gaussian_like(lam, n, seed=0):
    """n points from N(0, diag(lam)): the same covariance eigenvalues as the data. Euclidean robust prune
    does not change under rotation or translation, so the eigenvectors and mean are not needed."""
    X = np.random.default_rng(seed + 1).standard_normal((n, len(lam)), dtype=np.float32)
    X *= np.sqrt(lam).astype(np.float32)
    return X


def sample_spectrum(X, label):
    """Covariance eigenvalues and d_eff of X (e.g. the Gaussian's own sample spectrum)."""
    acc = accumulate(row_blocks(X), len(X), label, device="numpy")
    return spectrum(acc.covariance()), acc.value()


### Robust prune ###

def cuda_available():
    """True if the real CuPy imports and sees at least one CUDA device; prints why not otherwise."""
    try:
        import cupy
        if getattr(cupy, "_placeholder", False):
            print("GPU check: cupy in this process is the CPU-path placeholder; restart the kernel")
            return False
        n = cupy.cuda.runtime.getDeviceCount()
        if n == 0:
            print("GPU check: CuPy sees no CUDA devices")
        return n > 0
    except Exception as e:
        print(f"GPU check failed: {type(e).__name__}: {e}")
        return False


def load_prune(use_gpu):
    """(prune, augment, cupy or None) from utils.py. On the CPU path, utils.py's top-level CuPy and numba
    imports are satisfied by empty placeholder modules (marked _placeholder) if they fail to import."""
    if use_gpu:
        import cupy as cp
        if getattr(cp, "_placeholder", False):
            raise RuntimeError("this process holds the CPU-path placeholder for cupy: restart the kernel")
        from utils import batchedEuclideanRobustPrune as prune, precomputeAugMatrix as augment
        return prune, augment, cp
    try:
        import cupy, cupyx.scipy.spatial.distance  # noqa: F401  (utils.py needs them importable)
    except Exception:
        for name in ("cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.spatial", "cupyx.scipy.spatial.distance"):
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name]._placeholder = True
        sys.modules["cupyx.scipy.spatial.distance"].cdist = None
    try:
        import numba  # noqa: F401
    except ImportError:
        numba = sys.modules["numba"] = types.ModuleType("numba")
        numba.njit, numba.prange, numba._placeholder = (lambda *a, **k: (lambda f: f)), range, True
    from utils import batchedEuclideanRobustPruneCPU as prune, precomputeAugMatrixCPU as augment
    return prune, augment, None


def graph_paths(graph_dir, name, alpha=1.0):
    """Adjacency list and computed-sources files, named as in simulrun.py."""
    stem = f"{name}-alpha" + f"{alpha:g}".replace(".", "p") + "-euclidean"
    return Path(graph_dir) / f"adj-list-{stem}.txt", Path(graph_dir) / f"{stem}-computed.txt"


def build_graph(X, name, graph_dir, prune_fns, alpha=1.0, batch_size=512, seed=0, mininterval=0.1):
    """Robust-prune graph of X, appended to the adjacency list one batch at a time in a seeded shuffled
    order. Each line is "source [(neighbor, uncovered), ...]": the neighbors in the order the prune added
    them, each with the number of points still uncovered after that edge. Sources already in the computed
    file are skipped, so an interrupted run resumes; a finished graph is not recomputed. Returns the
    adjacency list path."""
    prune, augment, cp = prune_fns
    adj_path, computed_path = graph_paths(graph_dir, name, alpha)
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(map(int, computed_path.read_text().split())) if computed_path.exists() else set()
    order = np.random.default_rng(seed + 2).permutation(len(X))
    todo = [int(s) for s in order if int(s) not in done]
    print(f"{name}: {len(done):,} sources done, {len(todo):,} to go -> {adj_path}", flush=True)
    if not todo:
        return adj_path

    data = cp.asarray(X) if cp is not None else X
    X_aug = augment(data)
    for start in tqdm(range(0, len(todo), batch_size), unit="batch", desc=name, mininterval=mininterval):
        batch = todo[start : start + batch_size]
        neighborhoods = prune(batch, data, X_aug, alpha=alpha)
        # Whole batch at once; a crash loses at most one batch (computed is written after adj).
        with open(adj_path, "a") as adj, open(computed_path, "a") as comp:
            for source, hood in zip(batch, neighborhoods):
                adj.write(f"{source} {hood}\n")       # hood: [(neighbor, uncovered after it), ...]
            comp.write("".join(f"{source}\n" for source in batch))
    del data, X_aug
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    return adj_path


### Statistics ###

def read_graph(adj_path, uncovered=False):
    """{source: neighbor array} from an adjacency list, or with uncovered=True {source: (neighbors,
    uncovered counts)}. Reads "source [(neighbor, uncovered), ...]" lines and the older
    "source [neighbor, ...]" lines (uncovered counts -1). A source written twice (a run interrupted
    between writing a batch's lines and its computed list) keeps its last line."""
    hoods = {}
    with open(adj_path) as f:
        for line in f:
            source, rest = line.split(" ", 1)
            body = rest.strip()[1:-1]
            if "(" in body:
                pairs = np.fromstring(body.replace("(", "").replace(")", ""), dtype=np.int64, sep=",")
                nbrs, unc = pairs[0::2], pairs[1::2]
            else:
                nbrs = np.fromstring(body, dtype=np.int64, sep=",")
                unc = np.full(len(nbrs), -1, dtype=np.int64)
            hoods[int(source)] = (nbrs, unc) if uncovered else nbrs
    return hoods


def graph_stats(hoods, n, reciprocity=True):
    """Edge and degree statistics; in-degree and reciprocity only for a complete graph.
    Returns (stats dict, out-degree array)."""
    sources = np.fromiter(hoods, dtype=np.int64, count=len(hoods))
    out_deg = np.fromiter((len(h) for h in hoods.values()), dtype=np.int64, count=len(hoods))
    stats = {"sources": len(hoods), "edges": int(out_deg.sum()),
             "out mean": out_deg.mean(), "out median": np.median(out_deg),
             "out p99": np.percentile(out_deg, 99), "out max": out_deg.max()}
    if len(hoods) == n:
        dst = np.concatenate(list(hoods.values()))
        in_deg = np.bincount(dst, minlength=n)
        stats |= {"in median": np.median(in_deg), "in p99": np.percentile(in_deg, 99), "in max": in_deg.max(),
                  "in zero": int((in_deg == 0).sum())}
        if reciprocity:                               # share of edges u->v whose reverse v->u exists
            src = np.repeat(sources, out_deg)
            stats["reciprocity"] = np.isin(dst * n + src, src * n + dst).mean()
    return stats, out_deg


def summarize(name, graphs, spectra, n, d, graph_dir, reciprocity=True):
    """Results table (saved to <name>-results.csv) and {label: out-degree array}.
    graphs: {label: adjacency path}; spectra: {label: (eigenvalues, d_eff)}."""
    rows, out_degrees = [], {}
    for label, adj_path in graphs.items():
        lam, d_eff = spectra[label]
        stats, out_degrees[label] = graph_stats(read_graph(adj_path), n, reciprocity)
        s = spectrum_summary(lam)
        rows.append({"dataset": label, "n": n, "d": d, "d_eff": d_eff, "d_eff / d": d_eff / d,
                     "dims for 50%": s["dims for 50%"], "dims for 90%": s["dims for 90%"],
                     "dims for 99%": s["dims for 99%"], **stats})
    results = pd.DataFrame(rows)
    results.to_csv(Path(graph_dir) / f"{name}-results.csv", index=False)
    return results, out_degrees


### Reading runs back (the visualizer notebooks) ###

def save_meta(graph_dir, name, **info):
    """Write <name>-meta.json (n, d, distance, settings) so a run can be inspected without its data."""
    path = Path(graph_dir) / f"{name}-meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))


def find_runs(graph_dir):
    """Names of every run in graph_dir: anything with a <name>-spectrum.npy (written as soon as a run has
    loaded its data) or a <name>-results.csv (written when it finishes)."""
    gd = Path(graph_dir)
    spectra = {p.name.removesuffix("-spectrum.npy") for p in gd.glob("*-spectrum.npy")
               if not p.name.endswith("-gaussian-spectrum.npy")}
    results = {p.name.removesuffix("-results.csv") for p in gd.glob("*-results.csv")}
    return sorted(spectra | results)


def run_info(graph_dir, name, alpha=1.0):
    """What is saved for one run: meta, dataset spectrum (None if missing), results table (None until
    finished), and for each graph its adjacency path and number of sources done. n and d come from the
    meta file, the results table or the spectrum, and are None if none of them has it."""
    gd = Path(graph_dir)
    meta_path, results_path, spectrum_path = (gd / f"{name}-meta.json", gd / f"{name}-results.csv",
                                              gd / f"{name}-spectrum.npy")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    results = pd.read_csv(results_path) if results_path.exists() else None
    lam = np.load(spectrum_path) if spectrum_path.exists() else None
    n = meta.get("n") or (int(results["n"].iloc[0]) if results is not None else None)
    d = meta.get("d") or (len(lam) if lam is not None else int(results["d"].iloc[0]) if results is not None else None)
    graphs = {}
    for label, graph_name in (("dataset", name), ("gaussian", f"{name}-gaussian")):
        adj_path, computed_path = graph_paths(gd, graph_name, alpha)
        done = len(set(computed_path.read_text().split())) if computed_path.exists() else 0
        graphs[label] = (adj_path, done)
    return {"name": name, "meta": meta, "n": n, "d": d, "lam": lam, "results": results, "graphs": graphs}


def gaussian_spectrum(graph_dir, name, lam, n, seed=0):
    """The Gaussian's sample spectrum: saved by the pipeline, or regenerated from the dataset spectrum
    (needs n; n x d float32 in memory) and saved. None if n or the dataset spectrum is unknown."""
    path = Path(graph_dir) / f"{name}-gaussian-spectrum.npy"
    if path.exists():
        return np.load(path)
    if n is None or lam is None:
        return None
    lam_syn, _ = sample_spectrum(gaussian_like(lam, n, seed), "gaussian")
    np.save(path, lam_syn)
    return lam_syn


def coverage_curve(hoods, n):
    """Mean fraction of the other n - 1 points still uncovered after k edges, k = 1 .. max out-degree
    (a source whose neighborhood is complete counts as 0). hoods: read_graph(..., uncovered=True).
    None for graphs in the old format, which has no uncovered counts."""
    unc = [u for _, u in hoods.values()]
    if any(len(u) and u[0] < 0 for u in unc):
        return None
    lens = np.fromiter(map(len, unc), dtype=np.int64, count=len(unc))
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)     # edge index within its source
    return np.bincount(pos, weights=np.concatenate(unc)) / len(unc) / (n - 1)
