"""Minimal graph neural networks in pure PyTorch (no PyG / DGL).

Everything works on an `edge_index` tensor of shape [2, E] (source, target),
so the same message-passing code scales to graphs that don't fit a dense
adjacency matrix.

Layers:
    GCNConv   - Kipf & Welling, 2017  (symmetric-normalised propagation)
    SAGEConv  - Hamilton et al., 2017 (mean aggregator + root weight)
    GATConv   - Velickovic et al., 2018 (multi-head additive attention)
"""
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Data: Planetoid Cora (public split: 140 train / 500 val / 1000 test)
# --------------------------------------------------------------------------
def load_cora(root="data"):
    """Load Cora in the original Planetoid format (Yang et al., 2016)."""
    objs = []
    for name in ["x", "y", "tx", "ty", "allx", "ally", "graph"]:
        with open(f"{root}/ind.cora.{name}", "rb") as f:
            kw = {"encoding": "latin1"} if sys.version_info > (3, 0) else {}
            objs.append(pickle.load(f, **kw))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx = [int(l) for l in open(f"{root}/ind.cora.test.index")]
    test_sorted = np.sort(test_idx)

    feats = sp.vstack((allx, tx)).tolil()
    feats[test_idx, :] = feats[test_sorted, :]
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_sorted, :]

    n = feats.shape[0]
    src, dst = [], []
    for u, nbrs in graph.items():
        for v in nbrs:
            if u != v:
                src += [u, v]
                dst += [v, u]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)  # dedupe

    X = torch.tensor(np.asarray(feats.todense()), dtype=torch.float32)
    X = X / X.sum(1, keepdim=True).clamp(min=1)  # row-normalise bag-of-words
    Y = torch.tensor(labels.argmax(1), dtype=torch.long)

    train = torch.zeros(n, dtype=torch.bool); train[: y.shape[0]] = True
    val = torch.zeros(n, dtype=torch.bool); val[y.shape[0]: y.shape[0] + 500] = True
    test = torch.zeros(n, dtype=torch.bool); test[test_sorted] = True
    return X, Y, edge_index, train, val, test


# --------------------------------------------------------------------------
# Message-passing helpers
# --------------------------------------------------------------------------
def add_self_loops(edge_index, n):
    loops = torch.arange(n, device=edge_index.device).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


def degree(index, n):
    return torch.zeros(n, device=index.device).index_add_(0, index, torch.ones_like(index, dtype=torch.float))


def scatter_softmax(scores, index, n):
    """Softmax of `scores` [E, H] grouped by `index` (target node)."""
    H = scores.size(1)
    idx = index.unsqueeze(1).expand(-1, H)
    smax = torch.full((n, H), -1e9, device=scores.device).scatter_reduce(0, idx, scores, "amax", include_self=True)
    ex = (scores - smax[index]).exp()
    denom = torch.zeros(n, H, device=scores.device).index_add_(0, index, ex)
    return ex / (denom[index] + 1e-16)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
class GCNConv(nn.Module):
    """H' = D^-1/2 (A + I) D^-1/2 H W"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        src, dst = ei
        deg = degree(dst, n)
        norm = deg[src].rsqrt() * deg[dst].rsqrt()
        h = self.lin(x)
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * norm.unsqueeze(1))
        return out + self.bias


class SAGEConv(nn.Module):
    """h_v' = W_self h_v + W_nbr mean_{u in N(v)} h_u"""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_nbr = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n = x.size(0)
        agg = torch.zeros_like(x).index_add_(0, dst, x[src])
        agg = agg / degree(dst, n).clamp(min=1).unsqueeze(1)
        return self.lin_self(x) + self.lin_nbr(agg)


class GATConv(nn.Module):
    """Multi-head graph attention: e_uv = LeakyReLU(a^T [W h_u || W h_v])."""

    def __init__(self, d_in, d_out, heads=8, concat=True, dropout=0.6):
        super().__init__()
        self.h, self.d, self.concat, self.dropout = heads, d_out, concat, dropout
        self.lin = nn.Linear(d_in, heads * d_out, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, d_out))
        self.att_dst = nn.Parameter(torch.empty(1, heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out if concat else d_out))
        for p in (self.lin.weight, self.att_src, self.att_dst):
            nn.init.xavier_uniform_(p)
        self.last_attention = None  # kept for analysis

    def forward(self, x, edge_index):
        n = x.size(0)
        src, dst = add_self_loops(edge_index, n)
        h = self.lin(x).view(n, self.h, self.d)
        a_src = (h * self.att_src).sum(-1)  # [N, H]
        a_dst = (h * self.att_dst).sum(-1)
        e = F.leaky_relu(a_src[src] + a_dst[dst], 0.2)  # [E, H]
        alpha = scatter_softmax(e, dst, n)
        self.last_attention = (torch.stack([src, dst]), alpha.detach())
        alpha = F.dropout(alpha, self.dropout, self.training)
        msg = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros_like(h).index_add_(0, dst, msg)
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class MLP(nn.Module):
    """Graph-free baseline: same capacity, ignores edges."""

    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(d_in, d_hid), nn.Linear(d_hid, d_out), dropout

    def forward(self, x, edge_index, return_hidden=False):
        x = F.dropout(x, self.p, self.training)
        h = F.relu(self.l1(x))
        out = self.l2(F.dropout(h, self.p, self.training))
        return (out, h) if return_hidden else out


class GNN(nn.Module):
    """Stack of `layers` convs. `residual=True` adds skip connections between
    hidden layers, which is the standard fix for over-smoothing."""

    def __init__(self, kind, d_in, d_hid, d_out, layers=2, dropout=0.5, residual=False):
        super().__init__()
        self.kind, self.p, self.residual = kind, dropout, residual
        self.convs = nn.ModuleList()
        if kind == "gat":
            heads = 8
            self.convs.append(GATConv(d_in, d_hid, heads, concat=True))
            for _ in range(layers - 2):
                self.convs.append(GATConv(d_hid * heads, d_hid, heads, concat=True))
            self.convs.append(GATConv(d_hid * heads, d_out, heads=1, concat=False))
        else:
            Conv = {"gcn": GCNConv, "sage": SAGEConv}[kind]
            dims = [d_in] + [d_hid] * (layers - 1) + [d_out]
            for a, b in zip(dims[:-1], dims[1:]):
                self.convs.append(Conv(a, b))

    def forward(self, x, edge_index, return_hidden=False):
        h = x
        hidden = None
        for i, conv in enumerate(self.convs):
            h_in = h
            h = F.dropout(h, self.p, self.training)
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h) if self.kind == "gat" else F.relu(h)
                if self.residual and h_in.shape == h.shape:
                    h = h + h_in
                hidden = h
        return (h, hidden) if return_hidden else h


def build(kind, d_in, d_out, layers=2, residual=False):
    if kind == "mlp":
        return MLP(d_in, 64, d_out)
    if kind == "gat":
        return GNN("gat", d_in, 8, d_out, layers, dropout=0.6, residual=residual)
    return GNN(kind, d_in, 64 if kind == "gcn" else 64, d_out, layers, dropout=0.5, residual=residual)
