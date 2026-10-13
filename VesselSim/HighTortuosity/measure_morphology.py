"""Vessel morphology of binary vessel volumes, to compare generated overlays with murine and IXI seed maps.

Per volume, from the skeleton (same skeletonization as measure_density.py):
- radius: EDT of the mask sampled on skeleton voxels
- branches: skeleton split at junctions (voxels with 3+ skeleton neighbours). For each branch, the path length
  (sum of 26-neighbour steps) and, for branches with two endpoints, tortuosity = path length / endpoint distance
- branches and endpoints per 128^3, connected components per 128^3
Optionally saves max-projection panels of 128^3 crops centred on random vessel voxels (--panels).

Inputs are .npy masks (any nonzero is vessel) or NIfTI label maps (--label selects the vessel value).
All inputs are assumed to be on the same grid (0.5 mm for IXI seed maps, murine and generated overlays).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from measure_density import skeleton_of

P = 128
PCTS = [10, 25, 50, 75, 90]
MIN_TORTUOSITY_LENGTH = 10  # shorter branches are dominated by voxelization
NEIGHBOURS = np.ones((3, 3, 3))
# Half of the 26 neighbour offsets, so each voxel pair is counted once.
HALF_OFFSETS = [d for d in np.ndindex(3, 3, 3) if d > (1, 1, 1)]


def load_mask(path, label=None):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path) > 0
    import nibabel as nib
    data = np.asarray(nib.load(str(path)).dataobj)
    return data == label if label is not None else data > 0


def shifted_pairs(a, offset):
    # Views (a[x], a[x + d]) over all x where both are inside the array, for d = offset - 1.
    d = np.array(offset) - 1
    lo = [slice(max(0, -k), a.shape[i] - max(0, k)) for i, k in enumerate(d)]
    hi = [slice(max(0, k), a.shape[i] - max(0, -k)) for i, k in enumerate(d)]
    return a[tuple(lo)], a[tuple(hi)]


def branch_stats(skel):
    n_neighbours = ndi.convolve(skel.astype(np.uint8), NEIGHBOURS.astype(np.uint8), mode="constant") - 1
    junction = skel & (n_neighbours >= 3)
    branches = skel & ~junction
    labels, n = ndi.label(branches, structure=NEIGHBOURS)
    if n == 0:
        return np.zeros(0), np.zeros(0), 0
    length = np.zeros(n + 1)
    for offset in HALF_OFFSETS:
        a, b = shifted_pairs(labels, offset)
        same = (a > 0) & (a == b)
        length += np.bincount(a[same], minlength=n + 1) * np.linalg.norm(np.array(offset) - 1)
    length = length[1:]

    in_branch = ndi.convolve(branches.astype(np.uint8), NEIGHBOURS.astype(np.uint8), mode="constant") - 1
    ends = branches & (in_branch <= 1)
    idx = np.argwhere(ends)
    lab = labels[ends]
    order = np.argsort(lab, kind="stable")
    idx, lab = idx[order], lab[order]
    ids, first, count = np.unique(lab, return_index=True, return_counts=True)
    two = count == 2
    chord = np.linalg.norm(idx[first[two]] - idx[first[two] + 1], axis=1)
    path = length[ids[two] - 1]
    keep = (path >= MIN_TORTUOSITY_LENGTH) & (chord > 0)
    tortuosity = path[keep] / chord[keep]
    n_endpoints = int((skel & (n_neighbours <= 1)).sum())
    return length, tortuosity, n_endpoints


def pcts(x):
    x = np.asarray(x, dtype=float)
    return dict(zip(map(str, PCTS), np.percentile(x, PCTS).round(3).tolist())) if x.size else None


def measure_volume(job):
    path, label = job
    mask = load_mask(path, label)
    skel = skeleton_of(mask)
    blocks = mask.size / P ** 3
    radii = ndi.distance_transform_edt(mask)[skel].astype(np.float32)
    length, tortuosity, n_endpoints = branch_stats(skel)
    _, n_components = ndi.label(mask, structure=NEIGHBOURS)
    return {
        "file": Path(path).name, "shape": list(mask.shape), "vessel_vox": int(mask.sum()),
        "skeleton_vox": int(skel.sum()), "branches": int(length.size), "endpoints": n_endpoints,
        "components": int(n_components), "blocks_128": blocks,
        "_radii": radii, "_length": length, "_tortuosity": tortuosity,
    }


def summarize(vols):
    radii = np.concatenate([v.pop("_radii") for v in vols])
    length = np.concatenate([v.pop("_length") for v in vols])
    tort = np.concatenate([v.pop("_tortuosity") for v in vols])
    vessel = sum(v["vessel_vox"] for v in vols)
    skel = sum(v["skeleton_vox"] for v in vols)
    return {
        "n_volumes": len(vols),
        "radius_pcts": pcts(radii),
        "radius_mean": float(radii.mean()),
        "branch_length_pcts": pcts(length),
        "branch_length_weighted_median": float(np.median(np.repeat(length, np.maximum(1, length.astype(int))))),
        "tortuosity_pcts": pcts(tort),
        "vessel_vox_per_skeleton_vox": vessel / skel,
        # Densities per vessel content, so sparse whole-head seed maps and dense overlays are comparable.
        "branches_per_1000_skeleton_vox": 1000 * sum(v["branches"] for v in vols) / skel,
        "endpoints_per_1000_skeleton_vox": 1000 * sum(v["endpoints"] for v in vols) / skel,
        "components_per_1000_skeleton_vox": 1000 * sum(v["components"] for v in vols) / skel,
    }


def vessel_crop_panels(paths, label, out, n=4, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    for ax, path in zip(axes, rng.choice(paths, n, replace=len(paths) < n)):
        mask = load_mask(path, label)
        c = np.argwhere(mask)[rng.integers(mask.sum())]
        lo = np.clip(c - P // 2, 0, np.maximum(np.array(mask.shape) - P, 0))
        crop = mask[lo[0]:lo[0] + P, lo[1]:lo[1] + P, lo[2]:lo[2] + P]
        ax.imshow(crop.max(2).T, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.set_title(f"{Path(path).name[:22]}\n{100 * crop.mean():.2f}% vessel")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=80)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--glob", default="*.npy")
    parser.add_argument("--label", type=int, default=None, help="Vessel label value for NIfTI label maps")
    parser.add_argument("--max_volumes", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--out", default=None, help="Output JSON")
    parser.add_argument("--panels", default=None, help="Save a PNG of vessel-centred 128^3 crop projections")
    args = parser.parse_args()

    paths = sorted(Path(args.dir).glob(args.glob))
    if args.max_volumes:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_volumes).astype(int)]
    with Pool(min(args.n_workers, len(paths))) as pool:
        vols = pool.map(measure_volume, [(str(p), args.label) for p in paths])
    res = {"directory": args.dir, "label": args.label, "summary": summarize(vols), "volumes": vols}
    print(json.dumps(res["summary"], indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    if args.panels:
        vessel_crop_panels([str(p) for p in paths], args.label, args.panels)


if __name__ == "__main__":
    main()
