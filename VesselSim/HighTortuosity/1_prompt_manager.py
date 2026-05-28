import argparse
import json
import math
import random
import re
import time
from pathlib import Path
import yaml
from google import genai
from google.genai import types

def load_yaml_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_volume_center(volume_size):
    return [float(x) / 2.0 for x in volume_size]

def validate_and_correct_config(data, sim_id, _cfg,):
    logs: list[str] = []

    volume_size = _cfg["simulation"]["volume_size"]
    volume_center = get_volume_center(volume_size)
    margin = _cfg["simulation"]["margin_from_border"]
    cfg = _cfg["validation"]

    required_sections = ["anatomical_context", "root_geometry", "branching_model", "radius_model", "tortuosity_model"]
    for section in required_sections:
        if section not in data: return None, [f"{sim_id}: Missing critical section: {section}"]

    ### Root geometry
    rg = data["root_geometry"]
    required_keys = ["origin_voxel", "initial_direction", "initial_radius"]
    for key in required_keys:
        if key not in rg: return None, [f"{sim_id}: Missing root_geometry.{key}"]

    if len(rg["origin_voxel"]) != 3: return None, [f"{sim_id}: root_geometry.origin_voxel must have length 3"]
    if len(rg["initial_direction"]) != 3: return None, [f"{sim_id}: root_geometry.initial_direction must have length 3"]

    rg["origin_voxel"] = [max(0, min(float(volume_size[i]), float(rg["origin_voxel"][i]))) for i in range(3)]
    rg["initial_radius"] = max(float(cfg["root_geometry"]["min_initial_radius"]), float(rg["initial_radius"]),)

    near_border = any((rg["origin_voxel"][i] < margin) or (rg["origin_voxel"][i] > (volume_size[i] - margin)) for i in range(3))
    if near_border:
        to_center = [float(volume_center[i]) - float(rg["origin_voxel"][i]) for i in range(3)]
        norm = math.sqrt(sum(c * c for c in to_center)) + 1e-8
        rg["initial_direction"] = [c / norm for c in to_center]
        logs.append("Fixed start direction: set to center-facing unit vector")

    ### Branching model
    bm = data["branching_model"]
    bm_cfg = cfg["branching_model"]
    required_keys = ["max_depth", "bifurcation_probability", "bifurcation_depth_decay", "bifurcation_angle",
        "termination_factor", "segment_length_vox", "max_total_segments"]
    for key in required_keys:
        if key not in bm: return None, [f"{sim_id}: Missing branching_model.{key}"]

    if "min" not in bm["segment_length_vox"] or "max" not in bm["segment_length_vox"]:
        return None, [f"{sim_id}: branching_model.segment_length_vox must contain min and max"]

    bm["bifurcation_probability"] = max(float(bm_cfg["min_bifurcation_probability"]),
                                        min(float(bm_cfg["max_bifurcation_probability"]), float(bm["bifurcation_probability"])),)
    bm["termination_factor"] = max(float(bm_cfg["min_termination_factor"]),
                                   min(float(bm_cfg["max_termination_factor"]), float(bm["termination_factor"])),)
    bm["max_total_segments"] = int(bm["max_total_segments"])

    if int(bm["max_total_segments"]) > int(bm_cfg["max_total_segments"]):
        bm["max_total_segments"] = random.randint(5, int(bm_cfg["max_total_segments"]))
        logs.append("Clamped max_total_segments")

    if bm["segment_length_vox"]["min"] > bm["segment_length_vox"]["max"]:
        bm["segment_length_vox"]["min"], bm["segment_length_vox"]["max"] = (bm["segment_length_vox"]["max"], bm["segment_length_vox"]["min"])
        logs.append("Swapped inverted segment_length_vox min/max")

    ### Radius model
    rm = data["radius_model"]
    required_keys = ["min_radius_vox", "radius_decay_factor"]
    for key in required_keys:
        if key not in rm: return None, [f"{sim_id}: Missing radius_model.{key}"]
    rm["min_radius_vox"] = max(float(cfg["radius_model"]["min_radius_vox"]), float(rm["min_radius_vox"]),)

    # Tortuosity model
    tm = data["tortuosity_model"]
    required_keys = ["segment_max_curvature", "collision_jitter_scale"]
    for key in required_keys:
        if key not in tm:return None, [f"{sim_id}: Missing tortuosity_model.{key}"]

    return data, logs


def generate_with_retry(client, model_name, prompt, system_instruction, cfg,):
    retries = int(cfg["retry"]["retries"])
    base_delay = float(cfg["retry"]["base_delay_seconds"])
    generation_cfg = cfg["generation"]

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=float(generation_cfg["temperature"]),
                    top_p=float(generation_cfg["top_p"]),
                    max_output_tokens=int(generation_cfg["max_output_tokens"]),
                    response_mime_type=str(generation_cfg["response_mime_type"]),
                )
            )
            return response

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_delay * (2 ** attempt)
                print(f"    [429 Limit Hit] Cooling down for {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded for rate-limit errors.")

def get_next_start_index(output_dir):
    output_dir = Path(output_dir)
    pattern = re.compile(r"^vessel_(\d{3})\.json$")
    existing_ids = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                existing_ids.append(int(match.group(1)))
    return max(existing_ids) + 1 if existing_ids else 0

def load_text_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r") as f:
        return f.read()

def extract_json_from_response(raw_text):
    raw_text = raw_text.strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    clean_json_str = match.group(0) if match else raw_text
    return json.loads(clean_json_str)

def save_json(data, output_path):
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate procedural vessel parameter JSON files with Gemini.")
    parser.add_argument("--config", type=str, default="prompt_config.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    # Config values
    api_key = cfg["api"]["key_env_var"]
    if api_key == "GOOGLE_API_KEY":
        print("ERROR: Please set your Google Gemini API Key.")
        return
    client = genai.Client(api_key=api_key)
    model_name = cfg["api"]["model_name"]
    target_count = int(cfg["generation"]["target_count"])
    output_dir = Path(cfg["generation"]["output_dir"])
    random.seed(int(cfg["generation"]["random_seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    system_instruction_path = cfg["generation"]["system_instruction_path"]
    system_instruction = load_text_file(system_instruction_path)

    print(f"--- Starting Generation of {target_count} Parameter Files ---")
    print(f"Output directory: {output_dir}")
    print(f"Gemini model: {model_name}")

    start_idx = get_next_start_index(output_dir)
    successful_saves = 0

    for i in range(target_count):
        anatomy = random.choice(cfg["anatomy_types"])
        file_num = start_idx + i
        file_id = f"vessel_{file_num:03d}"

        try:
            response = generate_with_retry(client=client, model_name=model_name,
                prompt=f"Generate a unique, realistic configuration for: {anatomy}",
                system_instruction=system_instruction, cfg=cfg, )

            data = extract_json_from_response(response.text)
            valid_data, logs = validate_and_correct_config(data, file_id, cfg)

            if valid_data is None:
                print(f"[FAIL] {file_id}: Validation failed - {logs}")
                continue

            filename = output_dir / f"{file_id}.json"
            save_json(valid_data, filename)

            log_str = f" | Fixes: {logs}" if logs else ""
            print(f"[OK] {file_id}: {anatomy}{log_str}")
            successful_saves += 1

        except json.JSONDecodeError: print(f"[ERR] {file_id}: Invalid JSON format received from Gemini.")
        except Exception as e: print(f"[ERR] {file_id}: API or system error - {e}")
        time.sleep(float(cfg["generation"]["sleep_seconds"]))

    print(f"--- Complete. {successful_saves}/{target_count} parameter files generated. ---")

if __name__ == "__main__":
    main()