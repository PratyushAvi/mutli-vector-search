"""
Chamfer ranker for multi-vector embeddings.

With unit-normalised vectors, min_j dist(q_i, d_j) under 1 - cos (or squared L2 = 2 - 2cos)
is attained at max_j <q_i, d_j>, so ranking by Chamfer distance is the same as ranking by
Chamfer similarity:

    asymmetric (ColBERT MaxSim):  CH(Q, D) = sum_i max_j <q_i, d_j>
    symmetric:                    mean_i max_j <q_i, d_j> + mean_j max_i <q_i, d_j>

Rankers take padded sets X [n, L, dim] with a boolean mask [n, L] and expose
encode_queries / encode_docs (-> tuple of tensors) and score (-> [n_q, n_d], higher is better).
"""

import torch


class ChamferRanker:
    def __init__(self, symmetric=False):
        self.symmetric = symmetric

    @property
    def tag(self):
        return "chamfer-sym" if self.symmetric else "chamfer"

    def config(self):
        return {"ranker": "chamfer", "symmetric": self.symmetric}

    def encode_queries(self, X, mask):
        return X, mask

    def encode_docs(self, X, mask):
        return X, mask

    def score(self, queries, docs):
        Q, qm = queries  # [b, lq, dim], [b, lq]
        D, dm = docs     # [c, ld, dim], [c, ld]
        b, lq, dim = Q.shape
        c, ld, _ = D.shape
        sim = (Q.reshape(b * lq, dim) @ D.reshape(c * ld, dim).T).view(b, lq, c, ld)
        neg = torch.finfo(sim.dtype).min

        # Query -> doc: best doc vector for every query vector.
        q2d = sim.masked_fill(~dm.view(1, 1, c, ld), neg).amax(dim=3)  # [b, lq, c]
        q2d = torch.where(qm.unsqueeze(2), q2d, 0).float()
        if not self.symmetric:
            return q2d.sum(dim=1)

        # Doc -> query: best query vector for every doc vector.
        d2q = sim.masked_fill(~qm.view(b, lq, 1, 1), neg).amax(dim=1)  # [b, c, ld]
        d2q = torch.where(dm.unsqueeze(0), d2q, 0).float()
        return q2d.sum(dim=1) / qm.sum(1, keepdim=True) + d2q.sum(dim=2) / dm.sum(1)
