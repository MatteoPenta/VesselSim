import os
import argparse
import json
import re
from pathlib import Path
import yaml
import numpy as np
import random
import math
import time
from convert_npy_to_nifti import convert_npy_to_nifti

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def draw_sphere(volume, position, radius):
    x, y, z = position
    r = int(math.ceil(radius))
    x_min, x_max = max(0, int(x - r)), min(volume.shape[0], int(x + r + 1))
    y_min, y_max = max(0, int(y - r)), min(volume.shape[1], int(y + r + 1))
    z_min, z_max = max(0, int(z - r)), min(volume.shape[2], int(z + r + 1))

    if x_min >= x_max or y_min >= y_max or z_min >= z_max: return

    grid_x, grid_y, grid_z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
    dist_sq = (grid_x - x) ** 2 + (grid_y - y) ** 2 + (grid_z - z) ** 2
    mask = dist_sq <= radius ** 2
    volume[x_min:x_max, y_min:y_max, z_min:z_max][mask] = 1

def rasterize_graphs(graphs, volume_size):
    # Redraws canonical graphs with the builder's spheres. Each edge starts at its parent node, where the parent
    # (or the root sphere) already drew a sphere at least as large, so this reproduces the builder's mask exactly.
    volume = np.zeros(volume_size, dtype=np.uint8)
    for graph in graphs:
        for edge in graph["edges"]:
            for point, radius in zip(edge["points"], edge["radii"]):
                draw_sphere(volume, point, radius)
    return volume

def graph_length(graph):
    # Total centerline length: sum of polyline step lengths over all edges.
    return float(sum(np.linalg.norm(np.diff(np.asarray(e["points"]), axis=0), axis=1).sum() for e in graph["edges"]))

class VesselBuilder:
    def __init__(self, config, volume_size, confine_to_volume=False):
        # config is a parameter dict; it may be None in forest mode, where each tree brings its own.
        self.config = config
        self.confine_to_volume = confine_to_volume
        self._blocked_by_border = False
        self.volume_size = volume_size
        self.volume = np.zeros(volume_size, dtype=np.uint8)
        if config is not None:
            self._set_models(config)
        self._tree_id = None
        self._tree_order = None
        self.trees = []
        self.total_segments = 0
        self.root_position = None
        self.segments = []
        self._bifurcated = set()
        self._drop_reasons = {}

    def _set_models(self, params):
        self.root = params['root_geometry']
        self.branching = params['branching_model']
        self.radius = params['radius_model']
        self.tortuosity = params['tortuosity_model']

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    def _get_random_orthogonal(self, v):
        random_vec = np.random.randn(3)
        ortho = np.cross(v, random_vec)
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(v, np.array([1.0, 0.0, 0.0]))
        return self._normalize(ortho)

    def _rotate_vector(self, vector, axis, angle_deg):
        angle_rad = np.radians(angle_deg)
        vector = self._normalize(vector)
        axis = self._normalize(axis)
        return (vector * math.cos(angle_rad) +
                np.cross(axis, vector) * math.sin(angle_rad) +
                axis * np.dot(axis, vector) * (1 - math.cos(angle_rad)))

    def _get_sphere_mask_local(self, center_local, shape, radius):
        grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_sq = ((grid_x - center_local[0]) ** 2 +
                   (grid_y - center_local[1]) ** 2 +
                   (grid_z - center_local[2]) ** 2)
        return dist_sq <= radius ** 2

    def _inside_volume(self, pos, radius):
        # The whole sphere of the given radius around pos lies inside the volume.
        hi = np.array(self.volume_size) - 1 - radius
        return bool(np.all(pos >= radius) and np.all(pos <= hi))

    def _check_collision_volumetric(self, curr_pos, next_pos, radius):
        self._blocked_by_border = False
        if self.confine_to_volume and not self._inside_volume(next_pos, radius):
            self._blocked_by_border = True
            return True
        r = int(math.ceil(radius))
        x_min = max(0, int(min(curr_pos[0], next_pos[0]) - r))
        x_max = min(self.volume_size[0], int(max(curr_pos[0], next_pos[0]) + r + 1))
        y_min = max(0, int(min(curr_pos[1], next_pos[1]) - r))
        y_max = min(self.volume_size[1], int(max(curr_pos[1], next_pos[1]) + r + 1))
        z_min = max(0, int(min(curr_pos[2], next_pos[2]) - r))
        z_max = min(self.volume_size[2], int(max(curr_pos[2], next_pos[2]) + r + 1))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            return True
        sub_volume = self.volume[x_min:x_max, y_min:y_max, z_min:z_max]
        curr_local = curr_pos - np.array([x_min, y_min, z_min])
        next_local = next_pos - np.array([x_min, y_min, z_min])
        box_shape = sub_volume.shape

        mask_curr = self._get_sphere_mask_local(curr_local, box_shape, radius+4)
        mask_next = self._get_sphere_mask_local(next_local, box_shape, radius+3)
        mask_check = mask_next & (~mask_curr)

        if np.any(sub_volume[mask_check] > 0):
            return True
        return False

    def _draw_sphere(self, position, radius):
        draw_sphere(self.volume, position, radius)

    def _sphere_occupied(self, position, radius):
        r = int(math.ceil(radius))
        lo = np.maximum(0, np.floor(position - r).astype(int))
        hi = np.minimum(self.volume_size, np.floor(position + r).astype(int) + 1)
        sub = self.volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return bool(np.any(sub[self._get_sphere_mask_local(position - lo, sub.shape, radius)] > 0))

    def _split(self, parent_radius):
        # Returns ((radius_a, angle_a), (radius_b, angle_b)) for the two daughters, thicker first.
        # Without murray_exponent, both daughters get parent_radius * radius_decay_factor (original behaviour).
        gamma = self.radius.get("murray_exponent")
        if gamma is None:
            r = parent_radius * self.radius["radius_decay_factor"]
            radii = (r, r)
        else:
            # Murray's law r_p^g = r_a^g + r_b^g with asymmetry ratio alpha = r_b / r_a in (0, 1].
            ratio = self.radius.get("asymmetry_ratio", {"min": 1.0, "max": 1.0})
            # Smallest alpha that keeps the thin daughter at or above min_radius_vox, when possible.
            # The range is raised above asymmetry_ratio if needed, so a feasible split never loses its thin daughter.
            m = float(self.radius["min_radius_vox"]) / parent_radius
            alpha_floor = m / (1 - m ** gamma) ** (1 / gamma) if m < 2 ** (-1 / gamma) else 1.0
            alpha = random.uniform(max(ratio["min"], alpha_floor), max(ratio["max"], alpha_floor))
            r_a = parent_radius / (1 + alpha ** gamma) ** (1 / gamma)
            radii = (r_a, alpha * r_a)

        if self.branching.get("bifurcation_angle_model", "fixed") == "murray":
            # Murray's optimal branching angles: the thinner daughter deviates more from the parent direction.
            r0, ra, rb = parent_radius, radii[0], radii[1]
            cos_a = (r0 ** 4 + ra ** 4 - rb ** 4) / (2 * r0 ** 2 * ra ** 2)
            cos_b = (r0 ** 4 + rb ** 4 - ra ** 4) / (2 * r0 ** 2 * rb ** 2)
            angles = [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in (cos_a, cos_b)]
            jitter = float(self.branching.get("bifurcation_angle_jitter", 0.0))
            angles = [max(0.0, a + random.uniform(-jitter, jitter)) for a in angles]
        else:
            angles = [self.branching["bifurcation_angle"]] * 2
        return (radii[0], angles[0]), (radii[1], angles[1])

    def _drop(self, parent_id, reason):
        # In forest mode every tree has its own root key, since all roots have parent None.
        if parent_id is None and self._tree_id is not None:
            parent_id = ("root", self._tree_id)
        self._drop_reasons.setdefault(parent_id, []).append(reason)

    def build(self):
        start_pos = np.array(self.root['origin_voxel'], dtype=float)
        init_dir = np.array(self.root.get('initial_direction', [0, 0, 1]), dtype=float)
        if np.linalg.norm(init_dir) == 0: init_dir = np.array([0, 0, 1])
        start_dir = self._normalize(init_dir)
        self.root_position = start_pos.tolist()
        self.grow_tree(start_pos, start_dir, self.config)
        return self.volume

    def grow_tree(self, root_pos, root_dir, params, tree_id=None):
        # Grows one tree into the shared volume, so it avoids every vessel already drawn.
        # tree_id is None in single-tree mode, which keeps the original output format.
        self._set_models(params)
        self._tree_id = tree_id
        self.total_segments = 0
        start_pos = root_pos
        start_radius = self.root['initial_radius']

        self._draw_sphere(start_pos, start_radius)

        queue = [(start_pos, root_dir, start_radius, 0, None)]
        max_segments = self.branching['max_total_segments']

        while queue and  self.total_segments < max_segments:
            curr_pos, curr_dir, curr_radius, depth, parent_id = queue.pop(0)

            if depth > int(self.branching["max_depth"]):
                self._drop(parent_id, "max_depth")
                continue
            if curr_radius < float(self.radius["min_radius_vox"]):
                self._drop(parent_id, "min_radius")
                continue

            seg_len = random.randint(self.branching['segment_length_vox']['min'], self.branching['segment_length_vox']['max'])
            steps = int(seg_len)
            segment_retries = 6
            segment_succeeded = False

            seg_start_pos = curr_pos.copy()
            seg_start_dir = curr_dir.copy()
            bend_axis_base = self._get_random_orthogonal(seg_start_dir)
            max_turn_deg = max(0.0, float(self.tortuosity['segment_max_curvature']))
            turn_deg_base = random.uniform(-max_turn_deg, max_turn_deg)
            angle_jitter_scale = self.tortuosity['collision_jitter_scale']
            axis_jitter_angle = 5.0

            for attempt in range(segment_retries):
                test_pos = seg_start_pos.copy()
                test_dir = seg_start_dir.copy()

                turn_jitter = random.uniform(-max_turn_deg * angle_jitter_scale, max_turn_deg * angle_jitter_scale)
                turn_deg = turn_deg_base + turn_jitter

                bend_axis = bend_axis_base
                if axis_jitter_angle > 0:
                    wobble_axis = self._get_random_orthogonal(bend_axis_base)
                    bend_axis = self._rotate_vector(bend_axis_base, wobble_axis, random.uniform(-axis_jitter_angle, axis_jitter_angle))
                    bend_axis = self._normalize(bend_axis)

                path_positions = []
                path_dirs = []
                blocked = False

                for _ in range(steps):
                    next_dir = self._rotate_vector(test_dir, bend_axis, turn_deg)
                    next_dir = self._normalize(next_dir)
                    next_pos = test_pos + next_dir

                    if self._check_collision_volumetric(test_pos, next_pos, curr_radius):
                        blocked = True
                        break

                    path_positions.append(next_pos)
                    path_dirs.append(next_dir)
                    test_pos = next_pos
                    test_dir = next_dir

                if not blocked:
                    for p in path_positions:
                        self._draw_sphere(p, curr_radius)

                    curr_pos = test_pos
                    curr_dir = test_dir
                    segment_succeeded = True
                    break

            if not segment_succeeded:
                self._drop(parent_id, "border_blocked" if self._blocked_by_border else "segment_blocked")
                continue
            self.total_segments += 1

            seg_id = len(self.segments)
            segment = {
                "segment_id": seg_id,
                "parent_segment_id": parent_id,
                "depth": depth,
                "radius": float(curr_radius),
                "start_position": seg_start_pos.tolist(),
                "path_positions": [p.tolist() for p in path_positions],
            }
            if tree_id is not None:
                segment["tree_id"] = tree_id
                segment["tree_order"] = self._tree_order
            self.segments.append(segment)

            base_bif_prob = self.branching['bifurcation_probability']
            decay = self.branching['bifurcation_depth_decay']
            bif_prob = base_bif_prob * (decay ** depth)

            if random.random() < bif_prob:
                self._bifurcated.add(seg_id)
                rot_axis = self._get_random_orthogonal(curr_dir)
                (rad_a, angle_a), (rad_b, angle_b) = self._split(curr_radius)

                # Both daughters lie in the same plane, on opposite sides of the parent direction.
                for new_dir, new_rad in ((self._rotate_vector(curr_dir, rot_axis, angle_a), rad_a),
                                         (self._rotate_vector(curr_dir, rot_axis, -angle_b), rad_b)):
                    if not self._check_collision_volumetric(curr_pos, curr_pos + new_dir, new_rad):
                        queue.append((curr_pos, new_dir, new_rad, depth + 1, seg_id))
                    else:
                        self._drop(seg_id, "border_blocked" if self._blocked_by_border else "bifurcation_blocked")

            else:
                term_factor = self.branching['termination_factor']
                term_prob = 1.0 - math.exp(-term_factor*depth)

                if random.random() > term_prob:
                    pos_next = curr_pos + curr_dir
                    if not self._check_collision_volumetric(curr_pos, pos_next, curr_radius):
                        queue.append((curr_pos, curr_dir, curr_radius, depth, seg_id))
                    else:
                        self._drop(seg_id, "border_blocked" if self._blocked_by_border else "continuation_blocked")
                else:
                    self._drop(seg_id, "termination_probability")

        for entry in queue:
            self._drop(entry[4], "max_segments")

    def _erase_last_tree(self, volume_backup, n_segments, tree_id):
        self.volume = volume_backup
        self._bifurcated = {i for i in self._bifurcated if i < n_segments}
        self._drop_reasons = {k: v for k, v in self._drop_reasons.items()
                              if not (isinstance(k, int) and k >= n_segments) and k != ("root", tree_id)}
        del self.segments[n_segments:]

    def grow_forest(self, param_sets, target_length, forest_cfg):
        # Adds trees until their total centerline length reaches target_length, or a tree/failure limit is hit.
        # Roots cycle through a shuffled grid of cells, so any prefix of the trees is spread over the volume.
        size = np.array(self.volume_size, dtype=float)
        cells = np.array(forest_cfg["root_grid_cells"])
        margin = float(forest_cfg["root_margin_vox"])
        span = (size - 1 - 2 * margin) / cells
        cell_queue = []
        failed, attempts, total = 0, 0, 0.0
        while True:
            if total >= target_length:
                stop = "target_reached"; break
            if len(self.trees) >= forest_cfg["max_trees"]:
                stop = "max_trees"; break
            if failed >= forest_cfg["max_failed_roots"]:
                stop = "max_failed_roots"; break
            if not cell_queue:
                cell_queue = list(np.ndindex(*cells))
                random.shuffle(cell_queue)
            cell = np.array(cell_queue.pop())
            root_pos = margin + (cell + np.random.uniform(0, 1, 3)) * span
            root_dir = self._normalize(np.random.normal(size=3))
            params_file, params = random.choice(param_sets)
            tree_id = attempts
            attempts += 1

            radius = params["root_geometry"]["initial_radius"]
            too_close = any(np.linalg.norm(root_pos - np.array(t["root_position"])) < forest_cfg["min_root_spacing_vox"]
                            for t in self.trees)
            if too_close or self._sphere_occupied(root_pos, radius + 3):
                failed += 1
                continue

            backup, n_segments = self.volume.copy(), len(self.segments)
            self._tree_order = len(self.trees)
            self.grow_tree(root_pos, root_dir, params, tree_id)
            if len(self.segments) - n_segments < forest_cfg["min_tree_segments"]:
                self._erase_last_tree(backup, n_segments, tree_id)
                failed += 1
                continue

            graph = self.canonical_graph(tree_id)
            tree = {"tree_id": tree_id, "tree_order": self._tree_order, "root_position": root_pos.tolist(),
                    "root_direction": root_dir.tolist(), "parameters_file": params_file,
                    "n_segments": len(self.segments) - n_segments, "length": graph_length(graph)}
            self.trees.append(tree)
            total += tree["length"]
        self._tree_id = None
        return {"attempts": attempts, "failed_roots": failed, "length": total, "stop_reason": stop}

    def raw_tree(self):
        return {"root_position": self.root_position, "segments": self.segments}

    def canonical_graph(self, tree_id=None):
        # Nodes: root, bifurcations (with >= 1 accepted child) and terminal endpoints.
        # Edges: chains of continuation segments between nodes, with exact centerline points.
        # In forest mode, tree_id selects one tree.
        segments = self.segments if tree_id is None else [s for s in self.segments if s.get("tree_id") == tree_id]
        children = {}
        for seg in segments:
            children.setdefault(seg["parent_segment_id"], []).append(seg["segment_id"])

        if tree_id is None:
            root_position, root_key = self.root_position, None
        else:
            root_position, root_key = segments[0]["start_position"], ("root", tree_id)
        root = {"id": 0, "type": "root", "position": root_position}
        if None not in children:
            root["termination_reasons"] = self._drop_reasons.get(root_key, [])
        nodes = [root]
        edges = []

        pending = [(0, seg_id) for seg_id in children.get(None, [])]
        while pending:
            source, seg_id = pending.pop(0)
            first = self.segments[seg_id]
            points = [first["start_position"]]
            radii = [first["radius"]]
            depths = [first["depth"]]
            segment_ids = []

            while True:
                seg = self.segments[seg_id]
                n = len(seg["path_positions"])
                segment_ids.append(seg_id)
                points += seg["path_positions"]
                radii += [seg["radius"]] * n
                depths += [seg["depth"]] * n
                kids = children.get(seg_id, [])
                if seg_id in self._bifurcated or len(kids) != 1:
                    break
                seg_id = kids[0]

            node = {"id": len(nodes), "position": points[-1]}
            if kids:
                node["type"] = "bifurcation"
                node["n_children"] = len(kids)
                pending += [(node["id"], kid) for kid in kids]
            else:
                node["type"] = "terminal"
                node["termination_reasons"] = self._drop_reasons.get(seg_id, [])
            nodes.append(node)

            edges.append({
                "id": len(edges),
                "source": source,
                "target": node["id"],
                "segment_ids": segment_ids,
                "points": points,
                "radii": radii,
                "depths": depths,
            })

        return {"nodes": nodes, "edges": edges}

def next_attempt_for_base(output_dir: str, base: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    pat = re.compile(rf"^{re.escape(base)}_(\d{{2}})$")
    max_att = -1
    for fn in os.listdir(output_dir):
        m = pat.match(fn)
        if m and os.path.isdir(os.path.join(output_dir, fn)):
            max_att = max(max_att, int(m.group(1)))
    return max_att + 1

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def forest_targets(fc):
    # Per-preset targets: skeleton voxels per 128^3 from the reference, and the matching graph centerline length
    # per volume. calibration_factor is graph length per skeleton voxel.
    ref = json.loads(Path(fc["reference"]).read_text())["summary"]["skeleton_per_128"]["mean"]
    blocks = np.prod(np.array(fc["volume_size"], dtype=float) / 128)
    factor = fc["calibration_factor"] if fc["calibration_factor"] is not None else 1.0
    return {preset: {"fraction": frac, "skeleton_per_128": frac * ref, "length": frac * ref * blocks * factor}
            for preset, frac in fc["density_fractions"].items()}

def load_param_sets(parameters_dir):
    paths = sorted(Path(parameters_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"No parameter files found in: {parameters_dir}")
    return [(p.name, load_json(p)) for p in paths]

def build_forest(index, fc, param_sets, target_length):
    seed = fc["base_seed"] + index
    seed_everything(seed)
    builder = VesselBuilder(None, tuple(fc["volume_size"]), bool(fc.get("confine_to_volume", True)))
    info = builder.grow_forest(param_sets, target_length, fc)
    trees = [{**t, "seed": seed, "graph": builder.canonical_graph(t["tree_id"])} for t in builder.trees]
    return builder, trees, {**info, "seed": seed}

def forest_worker(job):
    index, fc, param_sets, targets, output_root = job
    t0 = time.time()
    top = max(targets, key=lambda k: targets[k]["fraction"])
    builder, trees, info = build_forest(index, fc, param_sets, targets[top]["length"])
    grow_seconds = time.time() - t0

    # Redrawing every tree from its graph must give back the grown mask, otherwise the prefixes would be wrong.
    full = rasterize_graphs([t["graph"] for t in trees], builder.volume_size) > 0
    if not np.array_equal(full, builder.volume > 0):
        raise RuntimeError(f"seg_{index:03d}: re-rasterized trees differ from the grown mask "
                           f"({int((full != (builder.volume > 0)).sum())} voxels)")

    cumulative = np.cumsum([t["length"] for t in trees])
    name = f"seg_{index:03d}"
    kept = {}
    for preset, target in targets.items():
        # Shortest prefix, in insertion order, whose length reaches the target (all trees if none does).
        # prefix_rule "closest" keeps one tree fewer when that lands nearer the target, removing the upward bias.
        k = min(int(np.searchsorted(cumulative, target["length"]) + 1), len(trees)) if preset != top else len(trees)
        if fc.get("prefix_rule", "reach") == "closest" and preset != top and k >= 2:
            if abs(cumulative[k - 2] - target["length"]) < abs(cumulative[k - 1] - target["length"]):
                k -= 1
        mask = full if k == len(trees) else rasterize_graphs([t["graph"] for t in trees[:k]], builder.volume_size) > 0
        out = Path(output_root) / preset
        np.save(out / "masks" / f"{name}.npy", mask)
        metadata = {
            "preset": preset, "fraction": target["fraction"],
            "target_skeleton_per_128": target["skeleton_per_128"], "target_length": target["length"],
            "achieved_length": float(cumulative[k - 1]) if k else 0.0,
            "n_trees_kept": k, "n_trees_grown": len(trees), "failed_roots": info["failed_roots"],
            "root_attempts": info["attempts"], "stop_reason": info["stop_reason"], "seed": info["seed"],
            "prefix_rule": fc.get("prefix_rule", "reach"),
            "calibration_factor": fc["calibration_factor"], "volume_shape": list(builder.volume_size),
            "coordinates": "continuous voxel indices (x, y, z) into the mask, i.e. mask[x, y, z]", "units": "voxels",
        }
        save_json(out / "graphs" / f"{name}_forest.json", {"metadata": metadata, "trees": trees[:k]})
        kept[preset] = k
    return {"index": index, "seconds": time.time() - t0, "grow_seconds": grow_seconds, "kept": kept, **info}

def run_forest(cfg, indices=None, output_root=None):
    from multiprocessing import Pool
    from measure_density import measure_dir, print_report

    fc = cfg["forest"]
    output_root = Path(output_root or fc["output_root"])
    param_sets = load_param_sets(fc.get("parameters_dir", cfg["parameters_dir"]))
    targets = forest_targets(fc)
    if fc["calibration_factor"] is None:
        print("calibration_factor is null: using 1.0 (graph length = skeleton voxels). Run --calibrate first.")
    print(f"calibration_factor = {fc['calibration_factor']} (graph length per skeleton voxel)")
    for preset, t in targets.items():
        print(f"  {preset}: target {t['skeleton_per_128']:.0f} skeleton vox / 128^3, {t['length']:.0f} vox centerline per volume")
        for sub in ("masks", "graphs"):
            (output_root / preset / sub).mkdir(parents=True, exist_ok=True)

    indices = list(range(fc["n_volumes"])) if indices is None else indices
    jobs = [(i, fc, param_sets, targets, str(output_root)) for i in indices]
    t0 = time.time()
    with Pool(min(fc["n_workers"], len(jobs))) as pool:
        results = pool.map(forest_worker, jobs)
    wall = time.time() - t0
    for r in results:
        print(f"  seg_{r['index']:03d}: {r['seconds']:.0f}s, trees kept {r['kept']}, failed roots {r['failed_roots']}, "
              f"stop {r['stop_reason']}")
    secs = [r["seconds"] for r in results]
    print(f"Wall clock {wall:.0f}s for {len(jobs)} volumes with {fc['n_workers']} workers; per volume "
          f"mean {np.mean(secs):.0f}s (min {np.min(secs):.0f}, max {np.max(secs):.0f})")
    save_json(output_root / "generation_log.json", {"forest_config": fc, "targets": targets, "wall_seconds": wall,
                                                    "volumes": results})

    for preset, t in targets.items():
        res = measure_dir(output_root / preset / "masks", n_workers=fc["n_workers"],
                          reference=fc["reference"], fraction=t["fraction"])
        save_json(output_root / preset / "density_report.json", res)
        print_report(res)
        print(f"  GPU cache for {preset}: {res['summary']['gpu_cache_mb']:.0f} MB "
              f"({res['summary']['n_volumes']} x 250^3 uint8)")

def calibrate_forest(cfg, n_volumes):
    # Graph length per skeleton voxel, measured on 1x forests grown with factor 1.0, on seeds outside the dataset.
    from multiprocessing import Pool
    fc = dict(cfg["forest"], calibration_factor=None)
    param_sets = load_param_sets(fc.get("parameters_dir", cfg["parameters_dir"]))
    target = max(forest_targets(fc).values(), key=lambda t: t["fraction"])["length"]
    with Pool(min(fc["n_workers"], n_volumes)) as pool:
        rows = pool.map(calibration_worker, [(10_000 + i, fc, param_sets, target) for i in range(n_volumes)])
    for length, skel in rows:
        print(f"  graph length {length:.0f}, skeleton voxels {skel}, ratio {length / skel:.4f}")
    factor = sum(r[0] for r in rows) / sum(r[1] for r in rows)
    print(f"calibration_factor = {factor:.4f} (graph length per skeleton voxel, {n_volumes} volumes)")

def calibration_worker(job):
    from measure_density import skeleton_of
    index, fc, param_sets, target = job
    builder, trees, _ = build_forest(index, fc, param_sets, target)
    return sum(t["length"] for t in trees), int(skeleton_of(builder.volume > 0).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="generation_config.yaml", help="Generation config, default is generation_config.yaml")
    parser.add_argument("--forest", action="store_true", help="Run forest mode even if forest.enabled is false")
    parser.add_argument("--calibrate", type=int, default=None, help="Forest mode: estimate calibration_factor on this many volumes")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Forest mode: only these volume indices")
    parser.add_argument("--output_root", type=str, default=None, help="Forest mode: override forest.output_root")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.forest or cfg.get("forest", {}).get("enabled", False):
        if args.calibrate:
            calibrate_forest(cfg, args.calibrate)
        else:
            run_forest(cfg, args.indices, args.output_root)
        return

    volume_size = tuple(cfg["initial_volume_size"])
    parameters_dir = Path(cfg["parameters_dir"])
    output_dir = Path(cfg["builder_output_dir"])
    masks_per_json = int(cfg["masks_per_json"])
    save_nifti = bool(cfg.get("save_nifti", False))
    confine_to_volume = bool(cfg.get("confine_to_volume", False))
    # Each sample gets its own seed, drawn from a master generator; a null seed gives a non-reproducible run.
    master_rng = random.Random(cfg.get("seed"))

    json_paths = sorted(parameters_dir.glob("*.json"))
    if len(json_paths) == 0:
        print(f"No parameter files found in: {parameters_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_paths)} JSON parameter files.")
    print(f"Generating {masks_per_json} vessel masks per JSON.")
    print(f"Saving masks and graphs to: {output_dir}")

    for json_path in json_paths:
        base = json_path.stem
        print(f"Processing {json_path.name}")
        params = load_json(json_path)

        for _ in range(masks_per_json):
            seed = master_rng.randrange(2 ** 32)
            try:
                seed_everything(seed)
                builder = VesselBuilder(params, volume_size, confine_to_volume)
                mask = builder.build()
                attempt = next_attempt_for_base(output_dir, base)
                sample_dir = output_dir / f"{base}_{attempt:02d}"
                sample_dir.mkdir()
                metadata = {
                    "parameters_file": json_path.name,
                    "parameters": params,
                    "seed": seed,
                    "confine_to_volume": confine_to_volume,
                    "volume_shape": list(volume_size),
                    "coordinates": "continuous voxel indices (x, y, z) into vessel_mask.npy, i.e. vessel_mask[x, y, z]",
                    "units": "voxels",
                }
                np.save(sample_dir / "vessel_mask.npy", mask)
                if save_nifti:
                    convert_npy_to_nifti(sample_dir / "vessel_mask.npy")
                save_json(sample_dir / "raw_tree.json", {"metadata": metadata, **builder.raw_tree()})
                save_json(sample_dir / "canonical_graph.json", {"metadata": metadata, **builder.canonical_graph()})
            except Exception as e:
                print(f"  Failed for {json_path.name}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()