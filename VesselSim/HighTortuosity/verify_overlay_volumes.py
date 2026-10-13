"""Acceptance checks for forest overlay volumes (output of 2_vessel_builder.py in forest mode).

- nesting: each lower-density mask is a subset of the next one, and redrawing every tree of the densest
  preset from its saved graph JSON reproduces that mask exactly
- hashes: SHA-256 of every mask, to compare two runs with the same seeds (--compare <other output_root>)
- graph-mask agreement (--agreement N): on the first N volumes of the densest preset, every skeleton component
  must lie within --tolerance voxels of a graph centerline point
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from measure_density import skeleton_of

spec = importlib.util.spec_from_file_location("vessel_builder", Path(__file__).with_name("2_vessel_builder.py"))
vessel_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vessel_builder)


def presets_by_density(root):
    presets = []
    for d in Path(root).iterdir():
        graphs = sorted((d / "graphs").glob("*_forest.json"))
        if graphs:
            presets.append((json.loads(graphs[0].read_text())["metadata"]["fraction"], d.name))
    return [name for _, name in sorted(presets)]


def mask_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(np.load(p).tobytes()).hexdigest()
            for p in sorted(Path(root).glob("*/masks/*.npy"))}


def load_trees(root, preset, name):
    return json.loads((Path(root) / preset / "graphs" / f"{name}_forest.json").read_text())["trees"]


def check_nesting(root, presets):
    densest = presets[-1]
    names = [p.stem for p in sorted((Path(root) / densest / "masks").glob("*.npy"))]
    subset_failures, raster_failures = [], []
    for name in names:
        masks = [np.load(Path(root) / p / "masks" / f"{name}.npy") for p in presets]
        for (lo_name, lo), (hi_name, hi) in zip(zip(presets, masks), zip(presets[1:], masks[1:])):
            if np.any(lo & ~hi):
                subset_failures.append(f"{name}: {lo_name} not within {hi_name}")
        trees = load_trees(root, densest, name)
        redrawn = vessel_builder.rasterize_graphs([t["graph"] for t in trees], masks[-1].shape) > 0
        if not np.array_equal(redrawn, masks[-1]):
            raster_failures.append(f"{name}: {int((redrawn != masks[-1]).sum())} voxels differ")
    return names, subset_failures, raster_failures


def check_agreement(root, preset, name, tolerance):
    mask = np.load(Path(root) / preset / "masks" / f"{name}.npy")
    trees = load_trees(root, preset, name)
    centerline = np.zeros(mask.shape, dtype=bool)
    for t in trees:
        for e in t["graph"]["edges"]:
            idx = np.clip(np.rint(np.asarray(e["points"])).astype(int), 0, np.array(mask.shape) - 1)
            centerline[tuple(idx.T)] = True
    dist = ndi.distance_transform_edt(~centerline)
    skel = skeleton_of(mask)
    labels, n = ndi.label(skel, structure=np.ones((3, 3, 3)))
    near = ndi.minimum(dist, labels, index=np.arange(1, n + 1)) if n else np.array([])
    return {"volume": name, "skeleton_components": int(n),
            "components_off_graph": int((near > tolerance).sum()),
            "skeleton_vox_within_tolerance": float((dist[skel] <= tolerance).mean()),
            "max_component_distance": float(near.max()) if n else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="overlay_volumes", help="Forest output_root")
    parser.add_argument("--compare", default=None, help="Second output_root generated with the same seeds")
    parser.add_argument("--agreement", type=int, default=2, help="Volumes for the graph-mask spot check")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Max skeleton-to-centerline distance (vox)")
    args = parser.parse_args()

    presets = presets_by_density(args.root)
    print(f"Presets, sparsest first: {presets}")
    names, subset_failures, raster_failures = check_nesting(args.root, presets)
    print(f"Nesting: {len(names)} volumes, subset failures {len(subset_failures)}, "
          f"re-rasterization failures {len(raster_failures)}")
    for line in subset_failures + raster_failures:
        print("  " + line)

    hashes = mask_hashes(args.root)
    print(f"Hashes: {len(hashes)} masks")
    if args.compare:
        other = mask_hashes(args.compare)
        common = sorted(set(hashes) & set(other))
        diff = [k for k in common if hashes[k] != other[k]]
        print(f"Reproducibility vs {args.compare}: {len(common)} masks compared, {len(diff)} differ")
        for k in diff:
            print("  differs: " + k)

    for name in names[:args.agreement]:
        print("Agreement:", check_agreement(args.root, presets[-1], name, args.tolerance))


if __name__ == "__main__":
    main()
