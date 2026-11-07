# graph-ml

Graph machine learning from first principles, then applied to a problem where graph structure is the whole
point. Both projects are plain PyTorch on an `edge_index` tensor (no PyG, no DGL), and every number comes from a
run in this repository.

| Folder | What it is | Headline result |
|---|---|---|
| [`gnn-from-scratch/`](gnn-from-scratch/) | GCN, GraphSAGE and GAT in about 200 lines, with message passing written out using `index_add_` and a scatter-softmax. Reproduced Cora benchmark (10 seeds), plus over-smoothing, label-efficiency and attention analyses. | GCN reaches 82.7% on Cora (paper: 81.5%). A 16-layer GCN collapses to 23% as embeddings converge; residual connections bring it back to 76%. |
| [`fraud-rings/`](fraud-rings/) | A fully synthetic generator for coordinated fraud rings (rare positives, weak per-account features, benign dense groups, camouflage edges), and a study of when graph features beat a GNN and when they beat tabular models. | Boosted trees on label-aware graph features reach 0.90 AUPRC, vs 0.55 for GraphSAGE and about 0.05 for tabular models alone. The README also documents a generator flaw found along the way: camouflage edges inflated fraud-node degree, which made detection easier, not harder. |

## Reading order

Start with `gnn-from-scratch` to see what a message-passing layer does, then `fraud-rings` to see graph signal
used on a task where the label is rare and the structure is adversarial.

## Run it

Each folder has its own `README.md` with exact commands and a `requirements.txt`. Cora is downloaded by the
project's data script; the fraud graph is generated locally. No real abuse data is used anywhere.

## License

MIT, see [LICENSE](LICENSE).
