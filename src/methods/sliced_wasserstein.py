"""
Sliced Wasserstein ranker for multi-vector embeddings.

Each item is treated as a uniform distribution over its vectors. For a unit direction
theta (a "slice"), the 1-D Wasserstein distance between projections is the L_p distance
between their quantile functions:

    W_p^p(theta#mu, theta#nu) = int_0^1 |F_mu^-1(t) - F_nu^-1(t)|^p dt

We fix L random slices (shared by queries and docs, seeded) and evaluate each quantile
function at K midpoints t_k = (k + 0.5) / K. Every set then maps to a single fixed-size
vector of L * K quantiles, and

    SW_p^p(Q, D) ~= mean_{l,k} |qhat_{l,k} - dhat_{l,k}|^p

For p = 2 this is a scaled squared Euclidean distance, so scoring is one matmul.
The score is -SW_p (higher is better). K is a quantile-grid approximation; it is exact
when K is a multiple of both set sizes.

Slices are nested: for a given seed, the first L slices are the same for every L, so
runs with 16, 64, 256 slices differ only in how many directions they average over.
"""

import torch


class SlicedWassersteinRanker:
    def __init__(self, dim, n_slices=64, n_quantiles=32, p=2, seed=0, device="cpu"):
        if p not in (1, 2):
            raise ValueError("p must be 1 or 2")
        if n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        self.dim, self.L, self.K, self.p, self.seed = dim, n_slices, n_quantiles, p, seed
        g = torch.Generator().manual_seed(seed)
        theta = torch.randn(n_slices, dim, generator=g)  # row l is slice l, independent of L
        self.theta = (theta / theta.norm(dim=1, keepdim=True)).T.contiguous().to(device)
        self.t = (torch.arange(n_quantiles, dtype=torch.float32, device=device) + 0.5) / n_quantiles

    @property
    def tag(self):
        return f"sw_L{self.L}_K{self.K}_p{self.p}_s{self.seed}"

    def config(self):
        return {"ranker": "sliced_wasserstein", "slices": self.L,
                "quantiles": self.K, "p": self.p, "seed": self.seed}

    def _quantiles(self, X, mask):
        """[n, l, dim] + mask -> flattened quantile embedding [n, L * K]."""
        P = X.float() @ self.theta                                   # [n, l, L]
        P = P.masked_fill(~mask.unsqueeze(2), float("inf")).sort(dim=1).values
        counts = mask.sum(dim=1, keepdim=True)                       # [n, 1]
        idx = (self.t.unsqueeze(0) * counts).long().clamp(max=P.shape[1] - 1)  # [n, K]
        q = P.gather(1, idx.unsqueeze(2).expand(-1, -1, self.L))     # [n, K, L]
        return q.transpose(1, 2).flatten(1)                          # slice-major [n, L * K]

    def encode_queries(self, X, mask):
        q = self._quantiles(X, mask)
        return q, (q * q).sum(1)

    def encode_docs(self, X, mask):
        return self.encode_queries(X, mask)

    def score(self, queries, docs):
        (q, qn), (d, dn) = queries, docs
        if self.p == 2:
            sq = (qn.unsqueeze(1) + dn.unsqueeze(0) - 2 * q @ d.T).clamp_min(0)
            return -(sq / q.shape[1]).sqrt()
        return -torch.cdist(q, d, p=1) / q.shape[1]
