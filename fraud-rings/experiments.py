"""Tabular vs graph-aware fraud detection on the synthetic ring graph.

    python experiments.py     # ~5 min on a laptop CPU
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud_graph import GraphConfig, make_graph, stratified_split

torch.set_num_threads(8)
MODELS = ["logreg", "gbm", "gbm_graph", "sage"]
LABEL = {"logreg": "Logistic regression", "gbm": "Gradient boosting",
         "gbm_graph": "GBM + graph features", "sage": "GraphSAGE"}
COLOR = {"logreg": "#9e9e9e", "gbm": "#ff7f0e", "gbm_graph": "#9467bd", "sage": "#1f77b4"}


# ---------------------------------------------------------------------------
# Graph features for the GBM: structure + neighbour aggregates + label propagation
# using ONLY training labels (no test-label leakage).
# ---------------------------------------------------------------------------
def neighbour_mean(v, ei, n):
    src, dst = ei
    s = torch.zeros(n, v.size(1)).index_add_(0, dst, v[src])
    d = torch.zeros(n).index_add_(0, dst, torch.ones(src.size(0))).clamp(min=1)
    return s / d.unsqueeze(1), d


def graph_features(X, ei, y, train):
    n = X.size(0)
    nbr, deg = neighbour_mean(X, ei, n)
    known = torch.zeros(n, 1); known[train, 0] = y[train].float()
    labelled = train.float().unsqueeze(1)
    src, dst = ei
    k_fraud = torch.zeros(n, 1).index_add_(0, dst, known[src])
    k_lab = torch.zeros(n, 1).index_add_(0, dst, labelled[src])
    frac_known_fraud = k_fraud / k_lab.clamp(min=1)
    # two-hop neighbour aggregate
    nbr2, _ = neighbour_mean(nbr, ei, n)
    return torch.cat([X, nbr, nbr2, deg.log1p().unsqueeze(1), k_fraud, frac_known_fraud], 1)


class SAGE(nn.Module):
    def __init__(self, d_in, d_hid=64, layers=2, p=0.3):
        super().__init__()
        dims = [d_in] + [d_hid] * layers
        self.self_l = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.nbr_l = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(d_hid, 1)
        self.p = p

    def forward(self, x, ei):
        n = x.size(0)
        for ls, ln in zip(self.self_l, self.nbr_l):
            agg, _ = neighbour_mean(x, ei, n)
            x = F.relu(ls(x) + ln(agg))
            x = F.dropout(x, self.p, self.training)
        return self.out(x).squeeze(1)


def fit_predict(kind, X, y, ei, train, val, seed):
    torch.manual_seed(seed)
    if kind == "logreg":
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[train], y[train])
        return m.predict_proba(X)[:, 1]
    if kind in ("gbm", "gbm_graph"):
        F_ = X if kind == "gbm" else graph_features(X, ei, y, train)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                           random_state=seed).fit(F_[train].numpy(), y[train].numpy())
        return m.predict_proba(F_.numpy())[:, 1]
    # GraphSAGE, full-batch, class-weighted BCE, early stopping on val AUPRC
    model = SAGE(X.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    pos_w = (y[train] == 0).sum() / (y[train] == 1).sum().clamp(min=1)
    best, best_p, bad = -1, None, 0
    for ep in range(400):
        model.train(); opt.zero_grad()
        logit = model(X, ei)
        loss = F.binary_cross_entropy_with_logits(logit[train], y[train].float(), pos_weight=pos_w)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(X, ei))
        ap = average_precision_score(y[val], p[val])
        if ap > best:
            best, best_p, bad = ap, p.clone(), 0
        else:
            bad += 1
            if bad > 40:
                break
    return best_p.numpy()


def metrics(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    r_at_p90 = float(rec[prec >= 0.9].max()) if (prec >= 0.9).any() else 0.0
    return {"auprc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "recall_at_p90": r_at_p90}


def run(cfg, train_frac, seeds, models=MODELS, keep_scores=False):
    X, y, ei, rid = make_graph(cfg)
    out = {k: [] for k in models}
    scores = {}
    for s in seeds:
        tr, va, te = stratified_split(y, train_frac, s)
        for k in models:
            p = fit_predict(k, X, y, ei, tr, va, s)
            out[k].append(metrics(y[te].numpy(), p[te.numpy()]))
            if keep_scores and s == seeds[0]:
                scores[k] = (y[te].numpy(), p[te.numpy()], rid[te].numpy())
    agg = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} |
              {m + "_std": float(np.std([r[m] for r in v])) for m in v[0]} for k, v in out.items()}
    return agg, scores, (X, y, ei, rid)


def main():
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})
    res = {}
    base = GraphConfig()

    # ---- A. main comparison (20% labelled) ----
    main_res, scores, (X, y, ei, rid) = run(base, 0.2, seeds=[0, 1, 2, 3, 4], keep_scores=True)
    res["dataset"] = {"accounts": int(X.size(0)), "fraud": int(y.sum()), "fraud_rate": float(y.float().mean()),
                      "rings": int(rid.max() + 1), "undirected_edges": int(ei.size(1) // 2),
                      "edge_homophily": float((y[ei[0]] == y[ei[1]]).float().mean()),
                      "fraud_edge_share_to_legit": float(((y[ei[0]] == 1) & (y[ei[1]] == 0)).sum() /
                                                          (y[ei[0]] == 1).sum())}
    res["main"] = main_res
    for k in MODELS:
        print(f"[main] {k:9} AUPRC={main_res[k]['auprc']:.3f} R@P90={main_res[k]['recall_at_p90']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k in MODELS:
        yt, p, _ = scores[k]
        pr, rc, _ = precision_recall_curve(yt, p)
        ax.plot(rc, pr, color=COLOR[k], label=f"{LABEL[k]} (AP {average_precision_score(yt, p):.2f})")
    ax.axhline(y.float().mean(), ls=":", color="k", lw=1, label="random")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, 20% of labels known", ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("figures/pr_curves.png"); plt.close(fig)

    # ---- B. ring-size recall at fixed alert budget ----
    budget = {}
    for k in ["gbm", "sage"]:
        yt, p, r = scores[k]
        k_alerts = int(yt.sum())
        flagged = np.zeros_like(yt); flagged[np.argsort(-p)[:k_alerts]] = 1
        sizes = np.bincount(rid[rid >= 0])
        bins = [(5, 10), (11, 20), (21, 30), (31, 40)]
        budget[k] = {}
        for lo, hi in bins:
            sel = (yt == 1) & (r >= 0) & (sizes[np.maximum(r, 0)] >= lo) & (sizes[np.maximum(r, 0)] <= hi)
            budget[k][f"{lo}-{hi}"] = float(flagged[sel].mean()) if sel.any() else float("nan")
    res["recall_by_ring_size"] = budget
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels_ = list(budget["gbm"].keys()); xs = np.arange(len(labels_))
    ax.bar(xs - .2, [budget["gbm"][l] for l in labels_], .4, color=COLOR["gbm"], label=LABEL["gbm"])
    ax.bar(xs + .2, [budget["sage"][l] for l in labels_], .4, color=COLOR["sage"], label=LABEL["sage"])
    ax.set_xticks(xs); ax.set_xticklabels(labels_)
    ax.set(xlabel="ring size", ylabel="recall", title="Recall by ring size (alerts = #fraud in test)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/recall_by_ring_size.png"); plt.close(fig)

    # ---- C. label scarcity ----
    fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    res["label_fraction"] = {}
    for f in fracs:
        r_, _, _ = run(base, f, seeds=[0, 1, 2])
        res["label_fraction"][f] = r_
        print(f"[labels] {f:.2f} " + " ".join(f"{k}={r_[k]['auprc']:.3f}" for k in MODELS))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in MODELS:
        ax.errorbar(fracs, [res["label_fraction"][f][k]["auprc"] for f in fracs],
                    yerr=[res["label_fraction"][f][k]["auprc_std"] for f in fracs],
                    marker="o", color=COLOR[k], label=LABEL[k], capsize=3)
    ax.set(xscale="log", xlabel="fraction of accounts with known label", ylabel="test AUPRC",
           title="How many labels do you need?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("figures/label_scarcity.png"); plt.close(fig)

    # ---- D. camouflage robustness ----
    cams = [0, 2, 5, 10, 20]
    res["camouflage"] = {}
    for c in cams:
        cfg = GraphConfig(camouflage=c)
        r_, _, (_, yy, ee, _) = run(cfg, 0.2, seeds=[0, 1, 2])
        r_["edge_homophily"] = float((yy[ee[0]] == yy[ee[1]]).float().mean())
        res["camouflage"][c] = r_
        print(f"[camo] {c:2} " + " ".join(f"{k}={r_[k]['auprc']:.3f}" for k in MODELS))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in MODELS:
        ax.plot(cams, [res["camouflage"][c][k]["auprc"] for c in cams], marker="o", color=COLOR[k], label=LABEL[k])
    ax.set(xlabel="avg camouflage edges per fraudster", ylabel="test AUPRC",
           title="Robustness to camouflage (fraud linking to legit users)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig("figures/camouflage.png"); plt.close(fig)

    # ---- E. what the data looks like ----
    G = nx.Graph(); G.add_edges_from(ei.T.tolist())
    rings = [0, 1, 2]
    nodes = set(torch.where(torch.isin(rid, torch.tensor(rings)))[0].tolist())
    for u in list(nodes):
        nodes |= set(list(G.neighbors(u))[:3]) if u in G else set()
    H = G.subgraph(nodes)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pos = nx.spring_layout(H, seed=1, k=0.25)
    col = ["#d62728" if y[v] == 1 else "#7f7f7f" for v in H.nodes]
    nx.draw_networkx_edges(H, pos, alpha=.25, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=col, node_size=22, ax=ax)
    ax.set_title("Three fraud rings (red) and their 1-hop neighbourhood (grey = legit)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig("figures/ring_example.png"); plt.close(fig)

    deg = torch.zeros(X.size(0)).index_add_(0, ei[1], torch.ones(ei.size(1)))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.arange(0, 40)
    ax.hist(deg[y == 0].numpy(), bins=bins, density=True, alpha=.6, label="legit", color="#7f7f7f")
    ax.hist(deg[y == 1].numpy(), bins=bins, density=True, alpha=.6, label="fraud", color="#d62728")
    ax.set(xlabel="degree (shared-entity links)", ylabel="density", title="Degree distribution")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/degree_distribution.png"); plt.close(fig)

    json.dump(res, open("results/results.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
