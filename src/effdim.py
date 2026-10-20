"""
Effective dimension (participation ratio) of large point sets, one pass and O(d^2) memory.

For centred points x_i with covariance C, PR = (tr C)^2 / tr(C^2) = (sum_k l_k)^2 / sum_k l_k^2:
1 if every point lies on one line, d if they are isotropic. `participation_ratio` computes the
U-statistic version that leaves out the i = j terms; `PRAccumulator` gives the same value from
running sums over blocks of points, so nothing is kept in memory and each file is read once.

Point sources (each yields float64 blocks of about `chunk` rows):
  row_blocks        one point per row of an (n, d) array, e.g. a single-vector vectors.npy
  item_mean_blocks  one point per item, the mean of its token vectors (encode.py multi-vector layout)

Both take optional sorted `items` indices, which `estimate_pr` uses for its sampling estimate.

The covariance spectrum (eigenvalues of the centred d x d covariance) comes from the same sums, so
`estimate_pr` returns it at no extra I/O: exact from a full pass, or from the sampled rows. With
m sampled rows and m >> d (e.g. 1e6 rows, d = 256) the sample spectrum is close to the full one;
for d/m not small, sample eigenvalues spread out (largest too large, smallest too small).

Used by effective_dim.ipynb (.npy datasets) and effective_dim_obliq.ipynb (OBLIQ, mean pooled).
Run `python src/effdim.py` for the sanity checks.
"""

import json
from pathlib import Path

import numpy as np

CHUNK = 1 << 16  # rows read per block


def participation_ratio(X):          # X: n x d, rows are points
    X = X - X.mean(0)
    n = len(X)
    sq = (X**2).sum(1)                                   # squared norms
    G2 = np.linalg.norm(X.T @ X, 'fro')**2 - (sq**2).sum()   # sum over i != j of <xi,xj>^2
    T2 = sq.sum()**2 - (sq**2).sum()                     # sum over i != j of |xi|^2 |xj|^2
    return T2 / G2                                       # the 1/(n(n-1)) factors cancel


class PRAccumulator:
    """participation_ratio over points added in blocks: one pass, O(d^2) memory, mergeable.

    Points are shifted by a fixed c (default: the first block's mean), y = x - c, which keeps
    float64 precise when the data has a large common mean. Keeps n, sum y, sum y y^T, sum |y|^2,
    sum |y|^4 and sum |y|^2 y. With delta = mean(y), x - mu = y - delta, so
      sum (x - mu)(x - mu)^T = sum y y^T - n delta delta^T
      sum |x - mu|^2         = sum |y|^2 - n |delta|^2
      sum |x - mu|^4         = sum (a - 2b + e)^2,  a = |y|^2, b = <y, delta>, e = |delta|^2
    """

    def __init__(self, shift=None):
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)
        self.n, self.a1, self.a2 = 0, 0.0, 0.0      # count, sum |y|^2, sum |y|^4
        self.s1 = self.S = self.m3 = 0.0            # sum y, sum y y^T, sum |y|^2 y

    def add(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.shift is None:
            self.shift = X.mean(0)
        Y = X - self.shift
        a = (Y * Y).sum(1)
        self.n += len(Y)
        self.s1 = self.s1 + Y.sum(0)
        self.S = self.S + Y.T @ Y
        self.m3 = self.m3 + a @ Y
        self.a1 += a.sum()
        self.a2 += (a * a).sum()
        return self

    def merge(self, other):
        """Add another accumulator's sums (same shift required, e.g. one per shard or worker)."""
        assert np.array_equal(self.shift, other.shift), "accumulators must share a shift"
        self.n += other.n
        self.s1, self.S, self.m3 = self.s1 + other.s1, self.S + other.S, self.m3 + other.m3
        self.a1 += other.a1
        self.a2 += other.a2
        return self

    def covariance(self):
        """Centred covariance matrix (d x d, divided by n - 1)."""
        delta = self.s1 / self.n
        return (self.S - self.n * np.outer(delta, delta)) / (self.n - 1)

    def value(self):
        n, delta = self.n, self.s1 / self.n
        e = delta @ delta
        C = self.S - n * np.outer(delta, delta)                 # sum of centred outer products
        sq_sum = self.a1 - n * e                                # sum |x - mu|^2
        b1, b2, ab = delta @ self.s1, delta @ self.S @ delta, delta @ self.m3
        sq2_sum = self.a2 + 4 * b2 + n * e * e - 4 * ab + 2 * e * self.a1 - 4 * e * b1   # sum |x - mu|^4
        G2 = (C * C).sum() - sq2_sum
        T2 = sq_sum**2 - sq2_sum
        return T2 / G2


def row_blocks(V, items=None, chunk=CHUNK):
    """Yield float64 rows of V (one point per row): all of them, or the sorted indices `items`."""
    n = len(V) if items is None else len(items)
    for s in range(0, n, chunk):
        idx = slice(s, s + chunk) if items is None else items[s : s + chunk]
        yield np.asarray(V[idx], dtype=np.float64)


def item_mean_blocks(V, offsets, items=None, chunk=CHUNK):
    """Yield float64 mean vectors of `items` (sorted indices; default every non-empty item) in blocks
    of about `chunk` token vectors. Item i's vectors are V[offsets[i]:offsets[i + 1]]."""
    if items is None:
        i, n = 0, len(offsets) - 1
        while i < n:
            j = min(n, max(i + 1, int(np.searchsorted(offsets, offsets[i] + chunk, side="right")) - 1))
            counts = np.diff(offsets[i : j + 1])
            keep = counts > 0
            if keep.any():
                X = np.asarray(V[offsets[i] : offsets[j]], dtype=np.float64)
                yield np.add.reduceat(X, offsets[i:j][keep] - offsets[i], axis=0) / counts[keep, None]
            i = j
        return
    starts, counts = offsets[items], offsets[items + 1] - offsets[items]
    starts, counts = starts[counts > 0], counts[counts > 0]
    ends = np.cumsum(counts)
    i = 0
    while i < len(counts):
        base = ends[i - 1] if i else 0
        j = min(len(counts), max(i + 1, int(np.searchsorted(ends, base + chunk, side="right"))))
        c = counts[i:j]
        local = np.r_[0, np.cumsum(c)[:-1]]                      # where each item starts in this block
        rows = np.repeat(starts[i:j] - local, c) + np.arange(c.sum())
        X = np.asarray(V[rows], dtype=np.float64)
        yield np.add.reduceat(X, local, axis=0) / c[:, None]
        i = j


def open_npy_dataset(folder):
    """Memory-mapped `<folder>/vectors.npy` (n x d). If `sq_norms.npy` exists it is checked against
    the first rows, to catch mismatched files. `ids.npy` is not needed and not read."""
    folder = Path(folder)
    V = np.load(folder / "vectors.npy", mmap_mode="r")
    if V.ndim != 2:
        raise ValueError(f"{folder}/vectors.npy: expected shape (n, d), got {V.shape}")
    norms = folder / "sq_norms.npy"
    if norms.exists():
        sq = np.load(norms, mmap_mode="r")
        head = np.asarray(V[:1000], dtype=np.float64)
        if len(sq) != len(V) or not np.allclose((head * head).sum(1), sq[: len(head)], rtol=1e-2, atol=1e-3):
            print(f"warning: {norms} does not match the squared norms of vectors.npy")
    return V


def open_multivector(prefix):
    """Memory-mapped vectors and offsets plus meta for `<prefix>.*` written by encode.py
    (skips ids.json, which is large at billion scale; encode.load_vectors also imports torch)."""
    V = np.load(f"{prefix}.vectors.npy", mmap_mode="r")
    offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")
    return V, offsets, json.loads(Path(f"{prefix}.meta.json").read_text())


def accumulate(blocks, total, label):
    """Feed blocks of points into a PRAccumulator (returned), printing progress."""
    acc, done, step = PRAccumulator(), 0, max(1, total // 20)
    for X in blocks:
        acc.add(X)
        if (done + len(X)) // step > done // step:
            print(f"\r  {label}: {done + len(X):,}/{total:,} items", end="", flush=True)
        done += len(X)
    print("\r" + " " * 80 + "\r", end="")
    return acc


def spectrum(cov):
    """Eigenvalues of a covariance matrix, largest first (tiny negative round-off clipped to 0)."""
    return np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)


def spectrum_summary(lam, levels=(0.5, 0.9, 0.99)):
    """Plug-in PR, share of variance in the top direction, and dimensions needed per variance level."""
    share = lam / lam.sum()
    cum = np.cumsum(share)
    return {"spectrum PR": lam.sum() ** 2 / (lam**2).sum(), "top-1 share": share[0],
            **{f"dims for {round(100 * q)}%": int(np.searchsorted(cum, q) + 1) for q in levels}}


def estimate_pr(n, blocks, label, exact=True, sample=None, seeds=5, seed=0):
    """PR and covariance spectrum of n points. `blocks(items)` yields point blocks (items=None: all n,
    else sorted indices).

    exact: one full pass ("PR"). sample: also estimate from `seeds` random subsets of that many points
    ("sample PR mean" / "sample PR sd"). If sample >= n there is nothing to sample, so the full pass
    runs even when exact is False.

    Returns (results dict, spectrum, spectrum sd). The spectrum is exact when the full pass ran, else
    the eigenvalues of the mean sample covariance, with the per-eigenvalue sd across samples.
    """
    sampling = bool(sample) and sample < n
    out, spec, spec_sd = {}, None, None
    if exact or not sampling:
        acc = accumulate(blocks(None), n, label)
        out["PR"] = acc.value()
        spec = spectrum(acc.covariance())
    if sampling:
        accs = []
        for s in range(seeds):
            items = np.sort(np.random.default_rng([seed, s]).choice(n, sample, replace=False))
            accs.append(accumulate(blocks(items), sample, f"{label} sample {s}"))
        est = [a.value() for a in accs]
        out |= {"sample size": sample, "sample PR mean": np.mean(est), "sample PR sd": np.std(est, ddof=1)}
        if spec is None:
            spec = spectrum(np.mean([a.covariance() for a in accs], axis=0))
            spec_sd = np.std([spectrum(a.covariance()) for a in accs], axis=0, ddof=1)
    print(f"{label}: " + "  ".join(f"{k}={v:.1f}" for k, v in out.items() if "PR" in k) + f"  ({n:,} points)")
    return out, spec, spec_sd


def _check():
    """Sanity checks on random data with a large common mean (as in anisotropic embeddings)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 64)) * np.linspace(0.1, 2, 64) + 50
    exact = participation_ratio(X)
    acc = PRAccumulator()
    for s in range(0, len(X), 777):
        acc.add(X[s : s + 777])
    assert np.isclose(acc.value(), exact)
    shift = X[:100].mean(0)
    assert np.isclose(PRAccumulator(shift).add(X[:2000]).merge(PRAccumulator(shift).add(X[2000:])).value(), exact)

    off = np.array([0, 3, 3, 10, 4000, 4001, 5000])   # includes an empty item and a 1-vector item
    means = np.array([X[a:b].mean(0) for a, b in zip(off[:-1], off[1:]) if b > a])
    assert np.allclose(np.concatenate(list(item_mean_blocks(X, off, chunk=500))), means)
    assert np.allclose(np.concatenate(list(item_mean_blocks(X, off, np.array([0, 1, 3, 5]), chunk=500))),
                       means[[0, 2, 4]])                # item 1 is empty and is dropped
    assert np.allclose(np.concatenate(list(row_blocks(X, chunk=777))), X)
    assert np.allclose(np.concatenate(list(row_blocks(X, np.array([2, 9, 4000]), chunk=2))), X[[2, 9, 4000]])
    # Spectrum: matches numpy's covariance; its plug-in PR matches the U-statistic up to O(1/n).
    assert np.allclose(acc.covariance(), np.cov(X, rowvar=False))
    lam = spectrum(acc.covariance())
    assert np.isclose(spectrum_summary(lam)["spectrum PR"], exact, rtol=1e-2)
    out, spec, sd = estimate_pr(len(X), lambda items: row_blocks(X, items, 500), "check", exact=False,
                                sample=2000, seeds=3)
    assert sd is not None and np.allclose(spec, lam, rtol=0.25)   # sample spectrum near the full one
    out, spec, sd = estimate_pr(len(X), lambda items: row_blocks(X, items, 500), "check", exact=False,
                                sample=10**6)                    # sample >= n: full pass instead
    assert "PR" in out and sd is None
    iso = participation_ratio(rng.standard_normal((20000, 64)))
    print(f"ok  anisotropic d=64: {exact:.2f}   isotropic d=64: {iso:.2f}")


if __name__ == "__main__":
    _check()
