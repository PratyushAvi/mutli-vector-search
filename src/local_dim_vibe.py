"""
Local effective dimension vs out-degree for VIBE datasets: the VIBE version of local_dim_pipeline.py, using the
robust-prune graphs that vibe_pipeline.py builds. effective_dim.ipynb reads the results together with those of
local_dim_pipeline.py (both write to results/local_dim/<name>/ in the same format).

For each VIBE dataset:
  1. the points the graph was built on, loaded exactly as vibe_pipeline.py loads them (vibe.load_train: the
     same --max-points / --seed subset, L2-normalised for cosine / angular datasets), so graph ids match rows
  2. the graph GRAPH_DIR/adj-list-<name>-alpha<a>-euclidean.txt (vibe.graph_paths); sources = the points it
     has a line for, out-degree = neighbours on that line
  3. local_dim_pipeline.compute: the exact K nearest neighbours of each source among all the points (the
     points are in RAM), and the local PR for every k and both centrings
and, unless --no-gaussian, the same for the Gaussian with the dataset's covariance spectrum and its graph
(<name>-gaussian), regenerated from <name>-spectrum.npy exactly as vibe_pipeline.py made it.

Writes OUT_DIR/<name>/ and OUT_DIR/<name>-gaussian/ (files as in local_dim_pipeline.py; meta.json has
source "vibe" and the global PR from the saved spectrum). <name> is the VIBE dataset name, plus
"-n<k>-seed<s>" with --max-points and "-alpha<a>" for alpha != 1.

Usage:
  python src/local_dim_vibe.py --dataset glove-200-cosine --dataset landmark-nomic-768-normalized \\
      --download-dir $SCRATCH/ANN-Search/datasets/vibe
  python src/local_dim_vibe.py --dataset glove-200-cosine --download-dir ... --max-points 200000 --max-sources 5000
"""

import argparse
from pathlib import Path

import numpy as np

import localdim
from local_dim_pipeline import add_compute_args, compute
from localdim import log
from paths import RESULTS_DIR
from vibe import gaussian_like, graph_paths, load_train, read_graph


def participation_ratio(lam):
    lam = np.asarray(lam, dtype=np.float64)
    return float(lam.sum() ** 2 / (lam**2).sum())


def graph_sources(adj, computed):
    """(sorted sources, out-degrees) from a vibe.py adjacency list, for the sources in its computed file."""
    hoods = read_graph(adj)
    done = set(map(int, computed.read_text().split()))
    sources = np.array(sorted(s for s in hoods if s in done), dtype=np.int64)
    if len(sources) < len(hoods):
        log(f"warning: {len(hoods) - len(sources):,} sources in {adj} are not in {computed}; left out")
    return sources, np.array([len(hoods[s]) for s in sources], dtype=np.int64)


def run_graph(label, graph_name, X, global_pr, info, args):
    """Local PR for one point set X (the VIBE points or the Gaussian) and its graph named graph_name."""
    adj, computed = graph_paths(args.graph_dir, graph_name, args.alpha)
    missing = [str(p) for p in (adj, computed) if not p.exists()]
    if missing:
        msg = f"{graph_name}: missing {', '.join(missing)} (run vibe_pipeline.py first)"
        if args.skip_missing:
            log(msg + "; skipped")
            return
        raise FileNotFoundError(msg)
    log(f"{graph_name}: reading graph {adj}")
    sources, out_deg = graph_sources(adj, computed)
    if len(sources) < len(X):
        log(f"{graph_name}: the graph is partial ({len(sources):,} of {len(X):,} sources computed)")
    out_name = graph_name + (f"-alpha{args.alpha:g}".replace(".", "p") if args.alpha != 1 else "")
    info = info | {"points": label, "adj": str(adj), "computed": str(computed), "alpha": args.alpha,
                   "global pr": global_pr}
    compute(out_name, localdim.ArrayFile(X), len(X), sources, out_deg, info, args)


def run_dataset(dataset, args):
    path = args.download_dir / f"{dataset}.hdf5"
    if not path.exists():
        msg = f"{dataset}: {path} not found (vibe_pipeline.py --download-only downloads it)"
        if args.skip_missing:
            log(msg + "; skipped")
            return
        raise FileNotFoundError(msg)
    log(f"{dataset}: loading {path}")
    X, name, attrs = load_train(path, args.max_points, args.data_seed)
    n, d = X.shape
    log(f"{name}: {n:,} x {d} float32 ({X.nbytes / 2**30:.1f} GiB), distance {attrs.get('distance')}")

    spectrum_path = args.graph_dir / f"{name}-spectrum.npy"
    lam = np.load(spectrum_path) if spectrum_path.exists() else None
    info = {"source": "vibe", "dataset": dataset, "file": str(path), "distance": attrs.get("distance"),
            "max points": args.max_points, "data seed": args.data_seed}
    run_graph("dataset", name, X, participation_ratio(lam) if lam is not None else None, info, args)
    del X

    if args.gaussian:
        if lam is None:
            log(f"{name}: no {spectrum_path}, so the Gaussian cannot be rebuilt; skipped")
            return
        syn_path = args.graph_dir / f"{name}-gaussian-spectrum.npy"
        syn_pr = participation_ratio(np.load(syn_path)) if syn_path.exists() else participation_ratio(lam)
        run_graph("gaussian", f"{name}-gaussian", gaussian_like(lam, n, args.data_seed), syn_pr, info, args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, help="VIBE dataset name, without .hdf5 (repeatable)")
    p.add_argument("--download-dir", required=True, type=Path, help="folder holding <dataset>.hdf5")
    p.add_argument("--graph-dir", type=Path, default=RESULTS_DIR / "vibe_graphs",
                   help="vibe_pipeline.py's --graph-dir (default: results/vibe_graphs)")
    p.add_argument("--max-points", type=int, help="as given to vibe_pipeline.py: the graph's random subset of points")
    p.add_argument("--data-seed", type=int, default=0, help="vibe_pipeline.py's --seed (subset and Gaussian)")
    p.add_argument("--alpha", type=float, default=1.0, help="vibe_pipeline.py's --alpha")
    p.add_argument("--no-gaussian", dest="gaussian", action="store_false",
                   help="skip the same-spectrum Gaussian and its graph")
    add_compute_args(p)
    args = p.parse_args()
    for dataset in args.dataset:
        run_dataset(dataset, args)


if __name__ == "__main__":
    main()
