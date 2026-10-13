"""Centerline density of human vessel labels on a 0.5 mm grid, comparable to the murine overlay patches.

For each subject: binarize, resample to 0.5 mm (linear + 0.5 threshold) unless already 0.5 mm,
skeletonize, then count centerline voxels in 128^3 crops centered on random vessel voxels
(the 80% vessel-weighted crops of the training pipeline).
"""
import argparse, csv, glob, json, time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

D = "/lustre/fswork/projects/rech/vnc/uzk93pr/data/"
SOURCES = {
    "IXI_VesselVerse": ("vesselFM/real_ixi/train/*Mask*.nii*", None),
    "TopCoW_MR_VesselVerse": ("TopCoWMRA_Manual_VesselVerse/labelsTr/*.nii*", lambda f: int(f[-10:-7]) < 131),
    "TubeTK_centerlines": ("TubeTK/tubeTK_segs_centerlines/*.nii*", None),
    "BraVa": ("BraVa/mask/*.nii*", None),
    "TopBrain_MR_arteries": ("TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_mr/*.nii*", None),
    "TopBrain_CT_arteries": ("TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_ct/*.nii*", None),
    "CAS": ("CAS/mask/*.nii*", None),
}
# Intracranial artery classes 1-34 (TopBrain v1); higher labels are extracranial arteries (MR) or veins (CT)
MAX_LABEL = {"TopBrain_MR_arteries": 34, "TopBrain_CT_arteries": 34}
P = 128

def cas_spacing():
    # CAS NIfTI headers carry unit spacing; the true spacing is in meta.csv
    rows = csv.DictReader(open(D + "CAS/meta.csv"))
    return {r["mask_path"].split("/")[-1]: tuple(float(x) for x in r["spacing"].strip("()").split(",")) for r in rows}

SPACING_OVERRIDE = {}

def load_05(f, max_label=None):
    img = nib.load(f)
    a = np.asarray(img.dataobj)
    a = (a > 0) & (a <= max_label) if max_label else a > 0
    z = np.array(SPACING_OVERRIDE.get(f.split("/")[-1], img.header.get_zooms()[:3]), dtype=float)
    if np.allclose(z, 0.5, atol=1e-3):
        return a
    return ndi.zoom(a.astype(np.float32), z / 0.5, order=1) > 0.5

def crop_counts(skel, vessel, n_crops, rng):
    idx = np.argwhere(vessel)
    out = []
    for c in idx[rng.choice(len(idx), n_crops)]:
        lo = np.clip(c - P // 2, 0, np.maximum(np.array(skel.shape) - P, 0))
        out.append(int(skel[lo[0]:lo[0]+P, lo[1]:lo[1]+P, lo[2]:lo[2]+P].sum()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_subjects", type=int, default=8)
    ap.add_argument("--n_crops", type=int, default=50)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--out", default="density_results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if "CAS" in args.sources:
        SPACING_OVERRIDE.update(cas_spacing())
    results = {}
    for name in args.sources:
        pat, keep = SOURCES[name]
        fs = [f for f in sorted(glob.glob(D + pat)) if keep is None or keep(f)]
        fs = [fs[i] for i in sorted(rng.choice(len(fs), min(args.n_subjects, len(fs)), replace=False))]
        rows = []
        for f in fs:
            t = time.time()
            v = load_05(f, MAX_LABEL.get(name))
            sk = skeletonize(v) > 0
            counts = crop_counts(sk, v, args.n_crops, rng)
            rows.append({"file": f.split("/")[-1], "vessel_frac": float(v.mean()),
                         "centerline_vox_total": int(sk.sum()),
                         "crop_centerline_mean": float(np.mean(counts)),
                         "crop_centerline_p10_p50_p90": np.percentile(counts, [10, 50, 90]).tolist()})
            print(name, rows[-1]["file"], f"total={rows[-1]['centerline_vox_total']} "
                  f"crop_mean={rows[-1]['crop_centerline_mean']:.0f} ({time.time()-t:.0f}s)", flush=True)
        results[name] = rows
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)

if __name__ == "__main__":
    main()
