"""
Local coverage curves from the local-prune files of local_dim_pipeline.py / local_dim_vibe.py (--local-prune).

For every result folder OUT_DIR/<name>/ holding local_prune_k<k>.txt files, writes
OUT_DIR/<name>/local_prune_coverage.csv: one row per (k, j) for j = 1 .. the largest local-prune degree, with
the mean, median and 90th percentile over sources of the fraction of N_k(v) still uncovered after v's first j
edges (0 once v is done) and the share of sources needing more than j edges. A few KB to a few hundred KB per
dataset, so it can go to git while the local_prune_k<k>.txt files (gitignored) stay on the cluster.
effective_dim.ipynb plots the curves from it.

The batch scripts write the same file when they finish; this rebuilds it for results made before that.

Usage:
  python src/local_prune_summary.py                       # every result in results/local_dim
  python src/local_prune_summary.py bigann yandex_deep    # only these
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

import localdim
from local_dim_pipeline import COVERAGE_CSV
from localdim import log
from paths import RESULTS_DIR


def summarize(rdir, overwrite=False):
    files = sorted(rdir.glob("local_prune_k*.txt"), key=lambda p: int(re.search(r"_k(\d+)\.txt$", p.name)[1]))
    if not files:
        return False
    out = rdir / COVERAGE_CSV
    if out.exists() and not overwrite and out.stat().st_mtime >= max(p.stat().st_mtime for p in files):
        log(f"{rdir.name}: {out.name} is up to date; skipped (--overwrite to rebuild)")
        return True
    t0, curves = time.time(), []
    for path in files:
        k = int(re.search(r"_k(\d+)\.txt$", path.name)[1])
        _, degree, unc = localdim.read_local_prune(path)
        curves.append(pd.DataFrame(localdim.coverage_summary(degree, unc, k)))
        log(f"{rdir.name}: k = {k}: {len(degree):,} sources, up to {degree.max():,} edges")
    df = pd.concat(curves)
    df.to_csv(out, index=False)
    log(f"{rdir.name}: wrote {out} ({len(df):,} rows, {out.stat().st_size / 1024:.0f} KB, {time.time() - t0:.0f} s)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="result folder names in OUT_DIR (default: all)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "local_dim",
                   help="the pipelines' --out-dir (default: results/local_dim)")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if the summary is newer than the files")
    args = p.parse_args()
    dirs = [args.out_dir / n for n in args.names] if args.names else sorted(d for d in args.out_dir.iterdir() if d.is_dir())
    done = [d.name for d in dirs if summarize(d, args.overwrite)]
    log(f"coverage summaries for {len(done)} result(s): {', '.join(done) or 'none (no local_prune_k*.txt files)'}")


if __name__ == "__main__":
    main()
