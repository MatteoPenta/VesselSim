# Graph viewer (Jean Zay)

View the generated vessel graphs overlaid on their masks, in your local browser.

SSH port forwarding (`ssh -L`) is blocked on Jean Zay, so the viewer is opened through the IDRIS JupyterHub proxy.

## 1. Generate samples (if needed)

```bash
cd /lustre/fswork/projects/rech/vnc/uzk93pr/projects/VesselSim/VesselSim/HighTortuosity
uv run python 2_vessel_builder.py
```

Each sample is written to `tmp/<name>/` with `vessel_mask.npy`, `raw_tree.json`, `canonical_graph.json`
(and `vessel_mask.nii.gz` if `save_nifti: true` in `generation_config.yaml`).

## 2. Start a JupyterLab session

Log in to https://jupyterhub.idris.fr and start a JupyterLab session (a login node is enough, no GPU needed).

## 3. Start the viewer

In a terminal **inside JupyterLab**:

```bash
cd /lustre/fswork/projects/rech/vnc/uzk93pr/projects/VesselSim/VesselSim/HighTortuosity
uv run python 5_visualize_graph.py --port 8050
```

Use `--dir <folder>` to view samples from another folder than `tmp/`.

## 4. Open it in your browser

Take your JupyterLab URL, keep everything before `/lab`, and add `/proxy/8050/`:

```
https://jupyterhub.idris.fr/user/<login>/<session>/proxy/8050/
```

Keep the trailing slash.

## 5. Stop

Press `Ctrl+C` in the terminal running the viewer.
