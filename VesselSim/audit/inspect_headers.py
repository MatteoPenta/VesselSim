"""Header summary (count, shape, spacing) and label values of each vessel label folder."""
import glob, sys
import numpy as np
import nibabel as nib

D = "/lustre/fswork/projects/rech/vnc/uzk93pr/data/"
FOLDERS = {
    "TubeTK tubeTK_segs_centerlines": "TubeTK/tubeTK_segs_centerlines/*.nii*",
    "TopBrain v1 MR": "TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_mr/*.nii*",
    "TopBrain v1 CT": "TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_ct/*.nii*",
    "CAS mask": "CAS/mask/*.nii*",
}
for name, pat in FOLDERS.items():
    fs = sorted(glob.glob(D + pat))
    zooms = np.array([nib.load(f).header.get_zooms()[:3] for f in fs])
    shapes = np.array([nib.load(f).shape[:3] for f in fs])
    labs = sorted(set().union(*[np.unique(np.asarray(nib.load(f).dataobj)).astype(int).tolist() for f in fs[:2]]))
    print(f"{name}: n={len(fs)} | spacing mm min {zooms.min(0).round(3)} max {zooms.max(0).round(3)} "
          f"| shape median {np.median(shapes,0).astype(int)} | labels(first 2) {labs[:20]}")
    sys.stdout.flush()
