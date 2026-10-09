import argparse
from pathlib import Path
import numpy as np
import nibabel as nib


def convert_npy_to_nifti(path):
    """Save a patch as NIfTI beside its .npy file, using unit voxel spacing."""
    path = Path(path)
    data = np.load(path)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D patch, got shape {data.shape}")
    out_path = path.with_suffix(".nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=str, required=True, help="Path to .npy patch")
    args = parser.parse_args()
    print(f"Saved: {convert_npy_to_nifti(args.patch)}")
