"""Synthetic account graph with coordinated fraud rings.

The generator is deliberately adversarial in the ways real abuse data is:
  * Fraud is rare (~3% of accounts) and organised in rings of 5-40 accounts
    that reuse a small pool of devices / payment instruments.
  * Per-account features only weakly separate fraud from legitimate users,
    so a tabular model alone struggles.
  * Legitimate users also form dense groups (households, shared office / NAT
    IPs), so "dense cluster" != "fraud".
  * Fraudsters add camouflage edges to random legitimate accounts.

No real data is used; every number here comes from the generator below.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphConfig:
    n_accounts: int = 20_000
    fraud_rate: float = 0.03
    ring_size: tuple = (5, 40)
    p_ring_edge: float = 0.25      # prob. two ring members share an entity
    camouflage: float = 2.0        # avg edges from each fraudster to random legit accounts
    n_households: int = 3_000      # legit groups of 2-4 sharing an address
    n_shared_ip: int = 150         # legit hubs of 10-60 accounts behind one IP
    feature_shift: float = 0.45    # mean shift of informative features for fraud
    n_features: int = 12
    n_informative: int = 4
    seed: int = 0


def make_graph(cfg: GraphConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_accounts
    y = np.zeros(n, dtype=np.int64)
    ring_id = -np.ones(n, dtype=np.int64)
    edges = []

    # --- fraud rings ------------------------------------------------------
    n_fraud_target = int(cfg.fraud_rate * n)
    perm = rng.permutation(n)
    ptr, r = 0, 0
    while ptr < n_fraud_target:
        size = int(rng.integers(cfg.ring_size[0], cfg.ring_size[1] + 1))
        members = perm[ptr: ptr + size]
        ptr += size
        y[members] = 1
        ring_id[members] = r
        # pairwise shared-entity edges, guarantee connectivity with a chain
        iu, ju = np.triu_indices(len(members), 1)
        keep = rng.random(len(iu)) < cfg.p_ring_edge
        edges += list(zip(members[iu[keep]], members[ju[keep]]))
        edges += list(zip(members[:-1], members[1:]))
        r += 1
    fraud = np.where(y == 1)[0]
    legit = np.where(y == 0)[0]

    # --- legitimate structure (confounders) -------------------------------
    for _ in range(cfg.n_households):
        g = rng.choice(legit, size=int(rng.integers(2, 5)), replace=False)
        iu, ju = np.triu_indices(len(g), 1)
        edges += list(zip(g[iu], g[ju]))
    for _ in range(cfg.n_shared_ip):
        g = rng.choice(n, size=int(rng.integers(10, 61)), replace=False)  # mixes some fraud in too
        hub = g[0]
        edges += [(hub, v) for v in g[1:]]                  # star: NAT gateway style
        iu, ju = np.triu_indices(len(g), 1)
        keep = rng.random(len(iu)) < 0.05
        edges += list(zip(g[iu[keep]], g[ju[keep]]))

    # --- camouflage -------------------------------------------------------
    n_cam = rng.poisson(cfg.camouflage, size=len(fraud))
    for u, k in zip(fraud, n_cam):
        for v in rng.choice(legit, size=k, replace=False):
            edges.append((u, v))

    # --- features ---------------------------------------------------------
    X = rng.normal(size=(n, cfg.n_features)).astype(np.float32)
    signs = rng.choice([-1, 1], size=cfg.n_informative)
    X[fraud, : cfg.n_informative] += cfg.feature_shift * signs
    # heavy-tailed nuisance feature (like order value) unrelated to label
    X[:, -1] = rng.lognormal(0, 1, size=n)
    X[:, -1] = (X[:, -1] - X[:, -1].mean()) / X[:, -1].std()

    e = np.array(edges, dtype=np.int64).T
    e = e[:, e[0] != e[1]]
    e = np.concatenate([e, e[::-1]], axis=1)
    e = np.unique(e, axis=1)
    return torch.tensor(X), torch.tensor(y), torch.tensor(e), torch.tensor(ring_id)


def stratified_split(y, train_frac, seed, val_frac=0.1):
    """Random stratified split; test = everything not in train/val."""
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    train = torch.zeros(n, dtype=torch.bool); val = torch.zeros(n, dtype=torch.bool)
    for c in (0, 1):
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        k_tr = max(1, int(train_frac * len(idx))); k_va = max(1, int(val_frac * len(idx)))
        train[idx[:k_tr]] = True
        val[idx[k_tr:k_tr + k_va]] = True
    test = ~(train | val)
    return train, val, test
