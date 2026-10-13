"""Centerline density of binary vessel volumes, as seen by vesselFM's RandVesselOverlayd.

For each .npy mask: vessel voxel fraction, skeleton voxels per 128^3, connected components, radius percentiles
(EDT sampled on the skeleton) and the skeleton-voxel distribution over random 128^3 crops placed like
RandVesselOverlayd (offset uniform in [0, size-128] per axis).
Skeletonization is the same as VesselSim/audit/vessel_density.py: skimage.morphology.skeletonize(mask) > 0.

Usage:
  python measure_density.py --dir <murine dir> --out density_reference_murine.json
  python measure_density.py --dir overlay_volumes/murine_0.5x/masks --reference density_reference_murine.json --fraction 0.5
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

P = 128
RADIUS_PCTS = [5, 25, 50, 75, 95]
CROP_PCTS = [5, 50, 95]


def skeleton_of(mask):
    return skeletonize(mask) > 0


def crop_counts(skel, n_crops, rng):
    # Skeleton voxels in random P^3 crops, using a 3D integral image.
    s = np.zeros(np.array(skel.shape) + 1, dtype=np.int64)
    s[1:, 1:, 1:] = skel.cumsum(0).cumsum(1).cumsum(2)
    hi = np.maximum(np.array(skel.shape) - P + 1, 1)
    lo = np.stack([rng.integers(0, hi[i], n_crops) for i in range(3)], axis=1)
    x0, y0, z0 = lo.T
    x1, y1, z1 = (lo + P).T
    return (s[x1, y1, z1] - s[x0, y1, z1] - s[x1, y0, z1] - s[x1, y1, z0]
            + s[x0, y0, z1] + s[x0, y1, z0] + s[x1, y0, z0] - s[x0, y0, z0])


def measure_volume(args):
    path, index, n_crops, seed = args
    t = time.time()
    mask = np.load(path) > 0
    skel = skeleton_of(mask)
    n_blocks = mask.size / P ** 3
    radii = ndi.distance_transform_edt(mask)[skel].astype(np.float32)
    _, n_components = ndi.label(mask, structure=np.ones((3, 3, 3)))
    counts = crop_counts(skel, n_crops, np.random.default_rng([seed, index]))
    return {
        "file": Path(path).name,
        "shape": list(mask.shape),
        "vessel_frac": float(mask.mean()),
        "skeleton_vox": int(skel.sum()),
        "skeleton_per_128": float(skel.sum() / n_blocks),
        "components": int(n_components),
        "radius_pcts": dict(zip(map(str, RADIUS_PCTS), np.percentile(radii, RADIUS_PCTS).round(3).tolist())),
        "crop_pcts": dict(zip(map(str, CROP_PCTS), np.percentile(counts, CROP_PCTS).tolist())),
        "seconds": round(time.time() - t, 1),
        "_crops": counts,
        "_radii": radii,
    }


def _indexed_measure(job):
    return job[1], measure_volume(job)


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std()), "cv": float(x.std() / x.mean()) if x.mean() else None,
            "min": float(x.min()), "max": float(x.max())}


def measure_dir(directory, n_crops=200, seed=0, n_workers=8, reference=None, fraction=None):
    paths = sorted(Path(directory).glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy masks in {directory}")
    jobs = [(str(p), i, n_crops, seed) for i, p in enumerate(paths)]
    vols = [None] * len(jobs)
    with Pool(min(n_workers, len(paths))) as pool:
        for i, v in pool.imap_unordered(_indexed_measure, jobs):
            vols[i] = v
            print(f"  measured {v['file']} ({v['seconds']}s)", flush=True)

    per_128 = [v["skeleton_per_128"] for v in vols]
    if reference is not None:
        ref_mean = json.loads(Path(reference).read_text())["summary"]["skeleton_per_128"]["mean"]
        target = {"reference": str(reference), "fraction": fraction, "skeleton_per_128": fraction * ref_mean}
    else:
        target = {"reference": "self", "fraction": 1.0, "skeleton_per_128": float(np.mean(per_128))}
    empty_thr = 0.1 * target["skeleton_per_128"]
    target["near_empty_threshold"] = empty_thr

    crops = np.concatenate([v.pop("_crops") for v in vols])
    radii = np.concatenate([v.pop("_radii") for v in vols])
    for v, c in zip(vols, np.split(crops, len(vols))):
        v["near_empty_frac"] = float((c < empty_thr).mean())

    summary = {
        "n_volumes": len(vols),
        "skeleton_per_128": stats(per_128),
        "deviation_from_target": float(np.mean(per_128) / target["skeleton_per_128"] - 1),
        "vessel_frac": stats([v["vessel_frac"] for v in vols]),
        "components": stats([v["components"] for v in vols]),
        "radius_pcts": dict(zip(map(str, RADIUS_PCTS), np.percentile(radii, RADIUS_PCTS).round(3).tolist())),
        "crop_pcts": dict(zip(map(str, CROP_PCTS), np.percentile(crops, CROP_PCTS).tolist())),
        "near_empty_frac": float((crops < empty_thr).mean()),
        "gpu_cache_mb": float(sum(np.prod(v["shape"]) for v in vols) / 1e6),  # uint8 cache in RandVesselOverlayd
    }
    settings = {"n_crops": n_crops, "crop_size": P, "seed": seed, "crop_offsets": "uniform in [0, size-128] per axis",
                "skeletonize": "skimage.morphology.skeletonize(mask) > 0", "components_connectivity": 26,
                "radius": "scipy EDT of mask sampled on skeleton voxels"}
    return {"directory": str(directory), "settings": settings, "target": target, "summary": summary, "volumes": vols}


def print_report(res):
    s, t = res["summary"], res["target"]
    print(f"{res['directory']}: {s['n_volumes']} volumes, target {t['skeleton_per_128']:.0f} skeleton vox / 128^3")
    for v in res["volumes"]:
        print(f"  {v['file']}: skel/128^3 {v['skeleton_per_128']:7.0f}  frac {100 * v['vessel_frac']:5.2f}%  "
              f"comp {v['components']:4d}  crops p5/p50/p95 {v['crop_pcts']['5']:.0f}/{v['crop_pcts']['50']:.0f}/"
              f"{v['crop_pcts']['95']:.0f}  near-empty {100 * v['near_empty_frac']:4.1f}%  ({v['seconds']}s)")
    k = s["skeleton_per_128"]
    print(f"  mean skel/128^3 {k['mean']:.0f} (std {k['std']:.0f}, CV {k['cv']:.3f}), deviation from target "
          f"{100 * s['deviation_from_target']:+.1f}%, vessel frac {100 * s['vessel_frac']['mean']:.2f}%, "
          f"near-empty crops {100 * s['near_empty_frac']:.1f}%, radius pcts {s['radius_pcts']}, "
          f"GPU cache {s['gpu_cache_mb']:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Directory of .npy masks")
    parser.add_argument("--out", default=None, help="Output JSON, default <dir>/../density_report.json")
    parser.add_argument("--reference", default=None, help="Reference JSON (e.g. density_reference_murine.json)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Target fraction of the reference mean")
    parser.add_argument("--n_crops", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()
    res = measure_dir(args.dir, args.n_crops, args.seed, args.n_workers, args.reference,
                      args.fraction if args.reference else None)
    out = Path(args.out) if args.out else Path(args.dir).parent / "density_report.json"
    out.write_text(json.dumps(res, indent=1))
    print_report(res)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
