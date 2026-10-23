"""
Run the VIBE pipeline end to end without plots (the batch version of vibe_robust_prune.ipynb):
download a VIBE dataset, compute its covariance spectrum, build a Gaussian with the same spectrum, build the
robust-prune graph of both, and write graph statistics. See vibe.py for the files written; the notebook
reads the same files, so it picks up graphs and results computed here.

Every step resumes: a complete download is reused, a partial one continues, and graph building skips sources
already done. Resubmitting the same command after a time limit continues where the last job stopped.

Usage:
  python src/vibe_pipeline.py --dataset landmark-nomic-768-normalized --download-dir $SCRATCH/vibe
  python src/vibe_pipeline.py --dataset glove-200-cosine --download-dir $SCRATCH/vibe --max-points 100000
  python src/vibe_pipeline.py --dataset yahoo-minilm-384-normalized --download-dir $SCRATCH/vibe --download-only
"""

import argparse
import time
from pathlib import Path

import numpy as np

from paths import RESULTS_DIR
from vibe import (build_graph, cuda_available, dataset_spectrum, describe, download, gaussian_like,
                  load_prune, load_train, sample_spectrum, save_meta, summarize)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="VIBE dataset name, without .hdf5")
    p.add_argument("--download-dir", required=True, type=Path, help="where the .hdf5 is (or will be) saved")
    p.add_argument("--graph-dir", type=Path, default=RESULTS_DIR / "vibe_graphs",
                   help="graphs, spectrum and results (default: results/vibe_graphs, as in the notebook)")
    p.add_argument("--max-points", type=int, help="use a seeded random subset of this many points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0, help="alpha-reachability, >= 1 (1: fully navigable)")
    device = p.add_mutually_exclusive_group()
    device.add_argument("--gpu", dest="use_gpu", action="store_true", default=None, help="require the GPU")
    device.add_argument("--cpu", dest="use_gpu", action="store_false", help="use NumPy even if a GPU exists")
    p.add_argument("--batch-size", type=int, help="sources pruned together (default: 512 on GPU, 64 on CPU)")
    p.add_argument("--no-verify", dest="verify", action="store_false", help="skip the download's SHA-256 check")
    p.add_argument("--no-reciprocity", dest="reciprocity", action="store_false",
                   help="skip reciprocity (needs ~24 bytes per edge)")
    p.add_argument("--download-only", action="store_true", help="download and stop (e.g. on a login node)")
    p.add_argument("--progress-every", type=float, default=60, help="seconds between progress lines")
    args = p.parse_args()
    if args.alpha < 1:
        p.error(f"--alpha must be >= 1, got {args.alpha}")
    t0 = time.time()

    path = download(args.dataset, args.download_dir, verify=args.verify, mininterval=args.progress_every)
    attrs, contents = describe(path)
    print(f"{path}\n  attributes: {attrs}\n  contents: {contents}", flush=True)
    if args.download_only:
        return

    if args.use_gpu is None:
        args.use_gpu = cuda_available()
    elif args.use_gpu and not cuda_available():
        p.error("--gpu given but CuPy sees no CUDA GPU (see the GPU check above)")
    batch_size = args.batch_size or (512 if args.use_gpu else 64)
    prune_fns = load_prune(args.use_gpu)
    print(f"prune: {prune_fns[0].__name__}  (batch size {batch_size})", flush=True)

    X, name, _ = load_train(path, args.max_points, args.seed)
    n, d = X.shape
    print(f"{name}: {n:,} x {d} float32 ({X.nbytes / 1e9:.2f} GB)", flush=True)
    save_meta(args.graph_dir, name, dataset=args.dataset, n=n, d=d, distance=attrs.get("distance"),
              point_type=attrs.get("point_type"), max_points=args.max_points, seed=args.seed, alpha=args.alpha,
              file=path)

    lam, d_eff = dataset_spectrum(X, name, args.graph_dir)
    X_syn = gaussian_like(lam, n, args.seed)
    lam_syn, d_eff_syn = sample_spectrum(X_syn, "gaussian")
    np.save(args.graph_dir / f"{name}-gaussian-spectrum.npy", lam_syn)
    print(f"d_eff = {d_eff:.1f} (d_eff / d = {d_eff / d:.3f}); Gaussian sample d_eff = {d_eff_syn:.1f}", flush=True)

    graphs = {}
    for label, data, graph_name in ((args.dataset, X, name), ("gaussian", X_syn, f"{name}-gaussian")):
        graphs[label] = build_graph(data, graph_name, args.graph_dir, prune_fns, args.alpha, batch_size,
                                    args.seed, mininterval=args.progress_every)
        print(f"{graph_name}: graph complete ({(time.time() - t0) / 3600:.2f} h elapsed)", flush=True)

    results, _ = summarize(name, graphs, {args.dataset: (lam, d_eff), "gaussian": (lam_syn, d_eff_syn)},
                           n, d, args.graph_dir, args.reciprocity)
    print(results.T.to_string(header=False))
    print(f"wrote {args.graph_dir / f'{name}-results.csv'}  ({(time.time() - t0) / 3600:.2f} h total)")


if __name__ == "__main__":
    main()
