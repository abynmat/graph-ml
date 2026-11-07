"""Reproducible experiments for gnn-from-scratch.

    python experiments.py            # runs everything, ~5 min on a laptop CPU

Writes JSON results to results/ and PNG figures to figures/.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from gnn import build, load_cora

torch.set_num_threads(8)
X, Y, EI, TRAIN, VAL, TEST = load_cora()
C = int(Y.max()) + 1
MODELS = ["mlp", "gcn", "sage", "gat"]
LABEL = {"mlp": "MLP (no graph)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
COLOR = {"mlp": "#9e9e9e", "gcn": "#1f77b4", "sage": "#2ca02c", "gat": "#d62728"}


def train_eval(kind, seed, train_mask=TRAIN, layers=2, residual=False, epochs=200, patience=50, log=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(kind, X.size(1), C, layers=layers, residual=residual)
    lr, wd = (0.005, 5e-4) if kind == "gat" else (0.01, 5e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_test, best_state, bad = 0, 0, None, 0
    curve = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(X, EI)[train_mask], Y[train_mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X, EI).argmax(1)
        va = (pred[VAL] == Y[VAL]).float().mean().item()
        te = (pred[TEST] == Y[TEST]).float().mean().item()
        if log:
            curve.append({"epoch": ep, "loss": loss.item(), "val": va, "test": te})
        if va > best_val:
            best_val, best_test, bad = va, te, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_val, best_test, model, curve


def smoothness(h):
    """Mean pairwise cosine similarity of node embeddings (1.0 = fully over-smoothed)."""
    h = F.normalize(h, dim=1)
    sim = h @ h.T
    n = h.size(0)
    return ((sim.sum() - n) / (n * (n - 1))).item()


def exp_benchmark(seeds=10):
    out = {}
    for k in MODELS:
        t = time.time()
        accs = [train_eval(k, s)[1] for s in range(seeds)]
        out[k] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "runs": accs,
                  "sec_per_run": (time.time() - t) / seeds}
        print(f"[benchmark] {k:5} {100*np.mean(accs):.1f} +- {100*np.std(accs):.1f}")
    return out


def exp_depth(depths=(2, 4, 8, 16), seeds=2):
    out = {}
    for res in (False, True):
        key = "gcn_residual" if res else "gcn_plain"
        out[key] = {}
        for L in depths:
            accs, sm = [], []
            for s in range(seeds):
                _, te, m, _ = train_eval("gcn", s, layers=L, residual=res)
                m.eval()
                with torch.no_grad():
                    _, h = m(X, EI, return_hidden=True)
                accs.append(te); sm.append(smoothness(h))
            out[key][L] = {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "cos_sim": float(np.mean(sm))}
            print(f"[depth] {key:13} L={L:2} acc={100*np.mean(accs):.1f} cos={np.mean(sm):.3f}")
    return out


def exp_label_rate(per_class=(1, 2, 5, 10, 20), seeds=3):
    """Fewer labels -> graph structure matters more."""
    out = {k: {} for k in MODELS}
    train_ids = torch.where(TRAIN)[0]
    for m in per_class:
        for k in MODELS:
            accs = []
            for s in range(seeds):
                g = torch.Generator().manual_seed(s)
                keep = []
                for c in range(C):
                    ids = train_ids[Y[train_ids] == c]
                    keep.append(ids[torch.randperm(len(ids), generator=g)[:m]])
                mask = torch.zeros_like(TRAIN); mask[torch.cat(keep)] = True
                accs.append(train_eval(k, s, train_mask=mask)[1])
            out[k][m] = {"acc": float(np.mean(accs)), "std": float(np.std(accs))}
            print(f"[labels] {k:5} {m:2}/class acc={100*np.mean(accs):.1f}", flush=True)
    return out


def main():
    torch.manual_seed(0)
    res = {"dataset": {"nodes": X.size(0), "edges_directed": EI.size(1), "features": X.size(1), "classes": C,
                       "homophily": float((Y[EI[0]] == Y[EI[1]]).float().mean())}}
    res["benchmark"] = exp_benchmark()
    res["depth"] = exp_depth()
    res["label_rate"] = exp_label_rate()

    # Training curves + embeddings + attention for one seed per model
    curves, hidden = {}, {}
    for k in MODELS:
        _, _, m, curve = train_eval(k, 0, log=True, patience=10**9, epochs=200)
        curves[k] = curve
        m.eval()
        with torch.no_grad():
            _, h = m(X, EI, return_hidden=True)
        hidden[k] = h
        if k == "gat":
            (ei, alpha) = m.convs[0].last_attention
    json.dump(res, open("results/results.json", "w"), indent=2)

    # ---------------- Figures ----------------
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Benchmark bar chart
    fig, ax = plt.subplots(figsize=(6, 3.6))
    b = res["benchmark"]
    xs = range(len(MODELS))
    ax.bar(xs, [100 * b[k]["mean"] for k in MODELS], yerr=[100 * b[k]["std"] for k in MODELS],
           color=[COLOR[k] for k in MODELS], capsize=4)
    for i, k in enumerate(MODELS):
        ax.text(i, 100 * b[k]["mean"] + 1.5, f"{100*b[k]['mean']:.1f}", ha="center")
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[k] for k in MODELS])
    ax.set_ylim(40, 90); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Cora node classification, 10 seeds (mean ± std)")
    fig.tight_layout(); fig.savefig("figures/benchmark.png"); plt.close(fig)

    # 2. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in MODELS:
        e = [c["epoch"] for c in curves[k]]
        axes[0].plot(e, [c["loss"] for c in curves[k]], color=COLOR[k], label=LABEL[k])
        axes[1].plot(e, [100 * c["val"] for c in curves[k]], color=COLOR[k], label=LABEL[k])
    axes[0].set(title="Training loss", xlabel="epoch", yscale="log")
    axes[1].set(title="Validation accuracy (%)", xlabel="epoch")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/training_curves.png"); plt.close(fig)

    # 3. Over-smoothing
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, ls, lab in [("gcn_plain", "-o", "GCN"), ("gcn_residual", "--s", "GCN + residual")]:
        d = res["depth"][key]; L = list(d.keys())
        axes[0].plot(L, [100 * d[l]["acc"] for l in L], ls, label=lab)
        axes[1].plot(L, [d[l]["cos_sim"] for l in L], ls, label=lab)
    axes[0].set(xscale="log", xlabel="layers", ylabel="test accuracy (%)", title="Accuracy vs depth")
    axes[1].set(xscale="log", xlabel="layers", ylabel="mean pairwise cosine sim", title="Over-smoothing of embeddings")
    for a in axes:
        a.set_xticks([2, 4, 8, 16]); a.set_xticklabels(["2", "4", "8", "16"])
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/oversmoothing.png"); plt.close(fig)

    # 4. Label efficiency
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k in MODELS:
        d = res["label_rate"][k]; m = list(d.keys())
        ax.errorbar(m, [100 * d[i]["acc"] for i in m], yerr=[100 * d[i]["std"] for i in m],
                    marker="o", color=COLOR[k], label=LABEL[k], capsize=3)
    ax.set(xscale="log", xlabel="labelled nodes per class", ylabel="test accuracy (%)",
           title="Label efficiency: graph models vs MLP")
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/label_efficiency.png"); plt.close(fig)

    # 5. t-SNE of hidden embeddings
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, k in zip(axes, ["mlp", "gat"]):
        z = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(hidden[k].numpy())
        ax.scatter(z[:, 0], z[:, 1], c=Y.numpy(), cmap="tab10", s=4)
        ax.set_title(f"t-SNE of hidden layer: {LABEL[k]}"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig("figures/tsne_embeddings.png"); plt.close(fig)

    # 6. GAT attention: does it attend more to same-class neighbours?
    src, dst = ei
    a = alpha.mean(1)
    not_self = src != dst
    same = (Y[src] == Y[dst]) & not_self
    diff = (Y[src] != Y[dst]) & not_self
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(a[same].numpy(), bins=bins, alpha=.6, density=True, label=f"same class (mean {a[same].mean():.3f})")
    ax.hist(a[diff].numpy(), bins=bins, alpha=.6, density=True, label=f"different class (mean {a[diff].mean():.3f})")
    ax.set(xlabel="attention weight (layer 1, head-averaged)", ylabel="density", title="GAT attention on Cora edges")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/gat_attention.png"); plt.close(fig)
    res["gat_attention"] = {"same_class_mean": a[same].mean().item(), "diff_class_mean": a[diff].mean().item()}
    json.dump(res, open("results/results.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
