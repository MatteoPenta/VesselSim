
This directory contains the segmentation model, training pipeline, and inference implementation, 
including reconstruction-based TTA. The final VesselSim pretrained model is available on [HuggingFace](https://huggingface.co/e-rain/VesselSim/tree/main).

A compatible PyTorch installation with CUDA support is recommended for training and inference.

Configuration files are located in `configs/`. The training and inference scripts use [Hydra](https://hydra.cc/), allowing configuration parameters to be modified directly in YAML files or overridden from the command line.

## Training

VesselSim is trained on synthetic 3D angiographic images and their corresponding vessel segmentation masks. The model uses a 3D DynUNet with an auxiliary reconstruction decoder and a training objective combining DiceCE, cbDice, and masked L1 reconstruction losses.

### 1. Prepare the training data

The training dataset should contain individual sample folders, each containing an image (`img.npy`) and its corresponding binary vessel segmentation mask (`mask.npy`).

Note that the our publicbly available data is separated by type and needs to be combined into one train directory to work as expect ([publicly available data](https://osf.io/8n5wu/overview))

The expected directory structure is:

```text
data/
├── train/
│   ├── 0/
│   │   ├── img.npy
│   │   └── mask.npy
│   ├── 1/
│   │   ├── img.npy
│   │   └── mask.npy
│   └── ...
└── val/
    ├── 0/
    │   ├── img.npy
    │   └── mask.npy
    └── ...
```

The model is configured to train on 3D patches of size `96 × 96 × 96`.

### 2. Configs
Search all #TODO in the config folders to set correct input and output paths from training, validation, and inference

### 3. Run training

From the SegmentationModel directory, run:

```bash
python train_recon.py
```

You can also override configuration parameters directly, ie:

```bash
python train_recon.py \
    lr=0.0001 \
    batch_size=8 \
    trainer.lightning_trainer.max_steps=46000
```


## Inference

VesselSim supports two inference modes:

1. **Zero-shot inference:** Direct segmentation using the pretrained model without any parameter updates.
2. **Reconstruction-based TTA:** Adaptation of the shared encoder to each input volume using a self-supervised masked reconstruction objective before segmentation.

Both modes use sliding-window inference to process 3D medical images.

### 1. Prepare the input images

Place the images to be segmented in a directory, for example:

```text
data/
└── test/
    └── images/
        ├── subject_001.nii.gz
        ├── subject_002.nii.gz
        └── ...
```

The default configuration expects NIfTI (`.nii.gz`) images. Input intensities are normalized using percentile-based scaling.

Images should be prepared with appropriate voxel spacing and orientation before inference. The supplied inference script does not automatically standardize physical voxel spacing.

### 2. Configure inference

Open `configs/inference_recon.yaml` and adjust #TODO and any other paramters for inference (augmentation, post processing. etc)

### 3. Zero-shot inference

To perform segmentation without test-time adaptation, set:

```yaml
tta_recon:
  apply: false
```

Then run:

```bash
python inference_recon.py
```

Alternatively, override the parameter directly:

```bash
python inference_recon.py tta_recon.apply=false
```


### 4. Reconstruction-based test-time adaptation

VesselSim also supports self-supervised adaptation to individual test volumes.

During TTA, random 3D patches are extracted from the input image and corrupted using coarse masking. The reconstruction decoder attempts to recover the masked image regions, and the resulting masked L1 reconstruction loss is used to update the shared encoder.

The segmentation and reconstruction decoders remain frozen during adaptation. After adaptation, the model performs segmentation on the original preprocessed image.

**Each input volume is adapted independently:** the model parameters are restored to their pretrained values before adapting to the next volume.



Run:

```bash
python inference_recon.py tta_recon.apply=true
```

Additional parameters can be overridden from the command line:

```bash
python inference_recon.py \
    tta_recon.apply=true \
    tta_recon.num_patches=24 \
    tta_recon.steps=1 \
    tta_recon.lr=1e-5
```

The number of sampled patches per adaptation round is controlled by `num_patches`, while `steps` controls the number of adaptation rounds.

### 5. Optional inference settings

The inference configuration also provides optional preprocessing, test-time augmentation, and post-processing operations:

- **Intensity inversion:** `tta.invert`
- **Histogram equalization:** `tta.equalize_hist`
- **Multi-scale inference:** `tta.scales`
- **Small connected-component removal:** `post.apply`
- **Prediction thresholding:** `merging.threshold`

These options can be adjusted depending on the input dataset. They should not be assumed to reproduce the published experiments unless the corresponding settings match those used for evaluation.

## Acknowledgment

This implementation was developed using [vesselFM](https://github.com/bwittmann/vesselFM) as its starting codebase.

In particular, vesselFM provided the initial **U-Net implementation, data-loading framework, training loop, and sliding-window inference scheme**, which were adapted and extended for VesselSim.

We thank the vesselFM authors for making their implementation publicly available.

If you use this repository, please also acknowledge the original vesselFM work:

```bibtex
@InProceedings{Wittmann_2025_CVPR,
    author    = {Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
    title     = {vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20874-20884}
}
```

## Citation

If you use VesselSim in your research, please cite:

```bibtex
@article{rainville2026vesselsim,
    title   = {VesselSim: learning 3D blood vessel segmentation without expert annotations},
    author  = {Rainville, Erin and Ananian, Melissa and Mirolla, Tristan and Rivaz, Hassan and Xiao, Yiming},
    journal = {arXiv preprint arXiv:2605.26277},
    year    = {2026}
}
```
