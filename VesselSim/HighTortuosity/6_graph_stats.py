import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def edge_table(graph):
    rows = []
    for e in graph["edges"]:
        pts = np.asarray(e["points"])
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        rows.append({
            "depth": e["depths"][0],
            "radius": e["radii"][0],
            "length": length,
            "tortuosity": length / chord if chord > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def sample_row(name, graph, mask):
    types = Counter(n["type"] for n in graph["nodes"])
    reasons = Counter(r for n in graph["nodes"] for r in n.get("termination_reasons", []))
    pts = np.asarray([p for e in graph["edges"] for p in e["points"]])
    edges = edge_table(graph)
    return {
        "sample": name,
        "nodes": len(graph["nodes"]),
        "bifurcations": types["bifurcation"],
        "one_child_bif": sum(1 for n in graph["nodes"] if n.get("n_children") == 1),
        "terminals": types["terminal"],
        "max_depth": int(edges["depth"].max()),
        "min_radius": round(float(edges["radius"].min()), 2),
        "border_terminals": reasons["border_blocked"],
        "inside": bool(np.all(pts >= 0) and np.all(pts <= np.array(mask.shape) - 1)),
        "occupancy_%": round(100 * float(mask.mean()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Summary statistics of canonical vessel graphs.")
    parser.add_argument("--dir", type=str, default="toy_samples", help="Sample folders directory, default is toy_samples")
    args = parser.parse_args()

    samples, edges, reasons = [], [], Counter()
    for d in sorted(p for p in Path(args.dir).iterdir() if (p / "canonical_graph.json").is_file()):
        graph = json.loads((d / "canonical_graph.json").read_text())
        mask = np.load(d / "vessel_mask.npy")
        samples.append(sample_row(d.name, graph, mask))
        reasons.update(r for n in graph["nodes"] for r in n.get("termination_reasons", []))
        e = edge_table(graph)
        e["family"] = graph["metadata"]["parameters_file"]
        edges.append(e)
    samples = pd.DataFrame(samples)
    edges = pd.concat(edges, ignore_index=True)

    pd.set_option("display.width", 200)
    print("Per sample:")
    print(samples.to_string(index=False))
    print("\nPer sample, summary:")
    print(samples.drop(columns=["sample", "inside"]).describe().round(1).to_string())

    print("\nTerminal reasons (all samples):", dict(reasons.most_common()))

    print("\nRadius by depth (all edges):")
    print(edges.groupby("depth")["radius"].agg(["count", "mean", "std", "min", "max"]).round(2).to_string())
    rho = spearmanr(edges["depth"], edges["radius"]).statistic
    print(f"\nSpearman(depth, radius) over edges: {rho:.3f}")

    print("\nPer parameter family:")
    print(edges.groupby("family")[["radius", "length", "tortuosity"]].agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()
