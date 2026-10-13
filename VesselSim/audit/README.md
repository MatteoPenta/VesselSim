# Human vessel data audit

Date: 2026-10-13. Covers what is on disk under `/lustre/fswork/projects/rech/vnc/uzk93pr/data/`.

- Counts, spacing and label values were measured with `inspect_headers.py`.
- Vessel density was measured with `vessel_density.py`; the raw numbers are in `density_results.json` and `density_log.txt`.
- Items marked **(verify)** are not confirmed from the files or from primary sources.

## Sources

**Test sets (MICCAI 2026).** Do not use these to train or calibrate a prior.

| Source | On disk | Spacing (mm) | Labels | Role |
|---|---|---|---|---|
| SMILE-UHURA | 14, `full_GT` | 0.3 iso (7T) | binary, expert | **test** |
| CereVessMRA, `AB*` | 121, `gt` | 0.26–0.49 × 0.8 | binary | **test** |
| TopCoW24 CT + VesselVerse | 35 (ct_131–165), `TopBrainCT_Manual_VesselVerse/labelsTr` | 0.36–0.60 × 0.5–0.75 | binary | **test** |

**Non-test sources.**

| Source | On disk | Spacing (mm) | Labels | Usable for a prior? |
|---|---|---|---|---|
| IXI + VesselVerse | 112, `vesselFM/real_ixi/train/*Mask*` | 0.5 iso (resampled) | binary, nnU-Net + manual refinement | yes, but these are the seed-map subjects |
| CereVessMRA, IXI sites (Guys/HH/IOP) | 150, `gt` | native | binary, CereVess protocol | yes; 74 subjects are among the 112 seed maps, 76 are not |
| TopCoW MR + VesselVerse | 125 (mr_001–165), `TopCoWMRA_Manual_VesselVerse/labelsTr` | 0.30–0.39 × 0.5–0.8 | binary | **only mr_001–130 (90 subjects)** |
| TopCoW 2024 CoW labels | 250 = 125 CT/MR pairs, `cow_seg_labelsTr` | 0.30–0.63 × 0.3–0.8 | 13-class, Circle of Willis only | CoW scaffold only; exclude pat_id ≥131 |
| TopBrain v1 (+ v2 TopAneu-36) | 25 MR + 25 CT, pat_id 001–027 (TopCoW patients) | MR 0.30–0.35 × 0.5–0.6; CT 0.40–0.55 × 0.3–0.75 | multiclass, **named segments**: 1–34 intracranial arteries (down to M3/A3/P3P4, SCA/AICA/PICA, AChA, OA); MR 35–42 extracranial arteries; CT 35–40 veins | yes. Binarize **labels 1–34 only** for an intracranial arterial prior |
| TubeTK, `tubeTK_segs_centerlines` | 97 | 0.513 × 0.513 × 0.8 | binary, rasterized official UNC centerline trees (single method) | **yes, use this one** |
| TubeTK, `00_MANUAL` (VesselVerse) | 109: the same 97 + 12 more | same | binary, mixed: rasterized UNC trees + nnU-Net | no, superseded |
| BraVa | 56 masks + 56 graphs | 0.56–0.63 × 0.62 | **graph-native**: SWC tree in RAS mm, about 3k points per subject, with `parent` and `radius`; mask rasterized from SWC | yes |
| CAS 2023 | 100 TOF-MRA, `mask` | 0.26–0.75 × 0.5–0.9 (**from `meta.csv`**; NIfTI headers wrongly say 1 mm) | binary; annotation protocol **(verify)** | yes; pathological: 85/100 have a stenosis box |

**Not on disk:** IXI centerlines from the VesselTok team (COSTA labels).

## Which TubeTK labels to use

Use `tubeTK_segs_centerlines` (97 subjects):
- **One observation process per source.** Under the Ambient framing, a quality value is attached to a source. Mixing rasterized trees with nnU-Net outputs inside one source would hide two different error processes behind a single label.
- **Graph-native origin.** These are rasterized UNC centerline + radius trees. The original trees can be used directly as graphs, like BraVa, with no skeletonization.
- **Little lost.** All 97 subjects are also in `00_MANUAL`, so dropping it loses only 12 subjects, which are nnU-Net-only. Their segmentation model's training data is unknown, which risks circularity.

## Leakage and overlaps

1. **TopCoW MR mr_131–165 are the same patients as the TopCoW24 CT test set.** The TopCoW release is 125 CT/MR pairs sharing `pat_id`, and all 35 CT test IDs (131–165) have MR VesselVerse labels and CoW labels. Using them would put the test patients' own vasculature into the prior. Exclude pat_id 131–165 everywhere.
2. **TopBrain (pat_id 001–027) does not overlap the test set.** Its patients are TopCoW patients 001–027, so they are the same subjects as TopCoW MR VesselVerse mr_001–027.
3. **The IXI-site CereVessMRA subjects are IXI subjects.** 74 of the 150 share an ID with the 112 seed-map subjects (matched by ID number, Guys016 ↔ IXI016). This doesn't affect the tests, since you already excluded them.

**Side effect: paired annotations of the same vessels** (useful for estimating q_clean empirically):
- IXI: VesselVerse vs CereVess, 74 subjects;
- TopCoW MR: VesselVerse vs TopBrain, 25 subjects.

## Vessel density on a 0.5 mm grid

Method: centerline voxels in 128³ crops centred on random vessel voxels, matching the 80% vessel-weighted training crops. 20 subjects per source (all 25 for TopBrain is about the same) and 50 crops per subject; test sets were not measured.

| Source | Centerline voxels per crop: mean (subject range) | Whole-volume centerline voxels (median) |
|---|---|---|
| IXI VesselVerse | 2,117 (1,304–3,366) | 9,620 |
| TubeTK centerlines | 1,751 (1,172–2,208) | 10,704 |
| BraVa | 1,496 (1,281–1,733) | 10,629 |
| TopCoW MR VesselVerse (≤130) | 1,476 (1,172–1,618) | 7,310 |
| CAS | 1,430 (1,023–2,237) | 4,264 |
| TopBrain MR, labels 1–34 | 1,042 (712–1,311) | 3,242 |
| TopBrain CT, labels 1–34 | 781 (580–984) | 2,724 |
| **Murine overlay** (uniform crop of 250³) | **≈3,700–4,000** | ≈28,000 per volume |

**Why TopBrain is sparsest:** only named segments are annotated (down to about M3/A3/P3P4). It is clean but truncated **by the labelling protocol**, not by the imaging, so a source can lose visibility through its protocol as well as its resolution.

## Answers to the three questions

**1. Oracle arm.** No fine, distal, non-test human source is on disk. Every non-test source is 1.5/3T-like at 0.26–0.75 mm. The only option is the SMILE split diagnostic (labels from some subjects as overlay, the others held out). It is not zero-shot and must be reported as a diagnostic.

**2. Real-data feasibility.**
- There are about 530 distinct non-test subjects with whole-brain labels: 112 IXI, 76 extra IXI-site CereVess, 90 TopCoW MR, 97 TubeTK, 56 BraVa, 100 CAS. TopBrain adds a second, anatomically labelled annotation of 25 TopCoW patients, plus their CT.
- Graph-native sources: BraVa (56) and TubeTK (97).
- **None of them is fine *and* clean.** A learned prior trained on this data alone would learn coarse vasculature. Fine-scale information still needs an extra source, e.g. unlabeled 7T TOF scans plus automatic segmentation (watch for circularity).

**3. Density target.**
- Visible human vessels give about 0.8–2.1k centerline voxels per vessel-centred crop. One murine overlay adds about 3.8k (before brain gating), so it multiplies the vessel content of a patch by roughly 2.5–5.
- A "human-like" density can't be calibrated from data, because all of these sources are lower bounds.
- Proposal: define the density arms as fractions of the murine density, **{1, 0.5, 0.25}× murine**, and don't claim a physiological value.
- Open caveat: gating to brain tissue removes part of each murine overlay, so its effective added density is below 3.8k.
