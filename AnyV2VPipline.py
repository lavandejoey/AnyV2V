#!/usr/bin/env python3
"""
V2V Pipeline - Colour Change Task (CSV-driven) using AnyV2V

- L40S ~6min/video

Usage:
    python3 AnyV2VPipline.py --csv /projects/hi-paris/DeepFakeDataset/DeepFake_V2/10k_real_video_captions_ziyi.csv --output_root /projects/hi-paris/DeepFakeDataset/DeepFake_V2/V2V/AnyV2V --anyv2v_root /home/infres/ziyliu-24/FakeParts2/StyleTrans/AnyV2V

Assumptions about CSV:
- "objects" column contains object names separated by commas/semicolons (e.g., "car, road sign")
- Optional colour hints may be present as "obj:colour" entries (e.g., "car:blue, road sign:yellow").
  If a colour is provided this way, we will respect it. Otherwise we auto‑assign distinct colours.
"""

import os
import sys
import json
import glob
import argparse
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from moviepy.editor import VideoFileClip

# ------------------------------ Defaults & Config ------------------------------

# Keep a conservative default set of distinct colour words (British spelling kept in file name).
DEFAULT_COLOUR_POOL = [
    "red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta",
    "teal", "pink", "gold", "silver"
]

BASE_PROMPT_SUFFIX = (
    "Edit the video realistically and smoothly while preserving style, motion, and scene. "
    "Avoid jitter, flicker, ghosting, compression artefacts, heavy blur, or pixelisation. "
    "Keep all non‑specified colours unchanged and the overall quality consistent with the input."
)

NEGATIVE_PROMPT = (
    "jitter, flicker, sudden scene jump, heavy blur, ghosting, compression artifacts, "
    "low resolution, pixelization on non‑sexual content, pixelization on faces"
)

# AnyV2V editing hyper‑parameters for colour change (kept from the draft with mild tweaks)
EDITING_PARAMS = {
    "ddim_init_latents_t_idx": 0,
    "pnp_f_t": 1.0,
    "pnp_spatial_attn_t": 1.0,
    "pnp_temp_attn_t": 1.0,
    "cfg": 9.0,
    "n_steps": 50,
}


# ------------------------------ Data classes ------------------------------

@dataclass
class CsvRow:
    name: str
    video_path: str
    caption: str
    # A list of (object, optional_colour) parsed from the "objects" column
    objects: List[Tuple[str, Optional[str]]]


@dataclass
class VideoInfo:
    video_name: str
    video_path: str
    duration: float
    fps: float
    size: Tuple[int, int]
    work_dir: str
    frames_dir: str
    demo_dir: str  # retained for compatibility


# ------------------------------ Utility helpers ------------------------------

def normalise_string(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip()


def safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", s)


def get_dist_info():
    """Read torchrun env; default to single-process if not present."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def restrict_visible_gpu_to_local_rank():
    """
    If CUDA_VISIBLE_DEVICES is a comma list, keep only the entry at LOCAL_RANK.
    Ensures each torchrun worker sees exactly one GPU.
    """
    _, local_rank, _ = get_dist_info()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd and "," in cvd:
        devs = [d.strip() for d in cvd.split(",") if d.strip() != ""]
        if 0 <= local_rank < len(devs):
            os.environ["CUDA_VISIBLE_DEVICES"] = devs[local_rank]


def parse_objects_column(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse the 'objects' column which may contain entries separated by commas or semicolons.
    It may optionally specify colours as 'object:colour'. Returns a list of (object, colour_or_None).
    """
    if not text or str(text).lower() == "nan":
        return []
    parts = re.split(r"[;,]", str(text))
    out: List[Tuple[str, Optional[str]]] = []
    for raw in parts:
        t = raw.strip()
        if not t:
            continue
        if ":" in t:
            obj, col = [x.strip() for x in t.split(":", 1)]
            if obj:
                out.append((obj, col if col else None))
        else:
            out.append((t, None))
    return out


def read_csv_like_diffusers(csv_path: Path) -> Dict[str, CsvRow]:
    """
    Reads CSV similarly to DiffusersV2V_CosmosPredict2.py:
    - Accepts abs_path | path | video | video_path
    - Optional name | caption
    - Required: a resolvable video path per row
    - Also reads 'objects' column for our colour‑change task
    Returns mapping name -> CsvRow
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    # Choose path column
    col_path = None
    for cand in ["abs_path", "path", "video", "video_path"]:
        if cand in df.columns:
            col_path = cand
            break
    if col_path is None:
        raise ValueError("CSV must include a column 'abs_path' (or 'path'/'video'/'video_path').")

    has_name = "name" in df.columns
    has_caption = "caption" in df.columns
    has_objects = "objects" in df.columns

    name_to_row: Dict[str, CsvRow] = {}

    for _, row in df.iterrows():
        p = str(Path(str(row[col_path])).expanduser())
        if not os.path.isfile(p):
            logging.warning(f"CSV path does not exist, skipping: {p}")
            continue
        name = str(row["name"]) if has_name and not pd.isna(row["name"]) else Path(p).stem
        caption = str(row["caption"]) if has_caption and not pd.isna(row["caption"]) else ""
        objects_list = parse_objects_column(str(row["objects"])) if has_objects else []

        name_to_row[name] = CsvRow(
            name=name,
            video_path=p,
            caption=caption,
            objects=objects_list
        )
    return name_to_row


def assign_two_distinct_colours(objs: List[Tuple[str, Optional[str]]]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """
    From a list of (object, maybe_colour), pick the first object and assign a colour.
    If colour hint is present, we use it; otherwise we pick from DEFAULT_COLOUR_POOL.
    Returns ((obj1, colour1),). Raises if no objects.
    """
    objs = [(o.strip(), (c.strip() if c else None)) for o, c in objs if o.strip()]
    if len(objs) < 1:
        raise ValueError("Need at least one object in the 'objects' column to perform colour change.")

    # Take first object only
    obj1, col1_hint = objs[0]

    # If hint exists, use it; otherwise pick from pool
    col1 = col1_hint or DEFAULT_COLOUR_POOL[0]

    return ((obj1, col1),)


def color_change_prompt(caption: str, obj1: str, colour1: str) -> str:
    """
    Build a single editing prompt that asks AnyV2V to recolour one object,
    while keeping other colours unchanged. Concatenate caption similarly to the Diffusers file.
    """
    caption = normalise_string(caption)
    main = (
        f"Change the colour of the {obj1} to {colour1}. "
        f"Keep all other elements and colours unchanged."
    )
    if caption and caption not in ("TIMEOUT", "nan"):
        return f"{caption}. {main} {BASE_PROMPT_SUFFIX}"
    else:
        return f"{main} {BASE_PROMPT_SUFFIX}"


def make_final_filename(name: str, obj1: str, col1: str) -> str:
    return f"{safe_slug(name)}_colourChange_{safe_slug(obj1)}2{safe_slug(col1)}.mp4"


def make_short_name(name: str, obj1: str, col1: str) -> str:
    """Generate short name without extension for image/video naming"""
    return f"{safe_slug(name)}_colourChange_{safe_slug(obj1)}2{safe_slug(col1)}"


# ------------------------------ AnyV2V Orchestration ------------------------------

class AnyV2VRunner:
    """
    Light wrapper around the AnyV2V command‑line utilities used in the user's draft.
    We keep the run_* methods similar to preserve compatibility.
    """

    def __init__(self, anyv2v_root: Path, output_root: Path):
        self.anyv2v_root = anyv2v_root
        self.output_root = output_root
        self.rank, self.local_rank, self.world_size = get_dist_info()
        self.i2vgen_xl_path = self.anyv2v_root / "i2vgen-xl"
        self.demo_path = self.anyv2v_root / "demo"
        self.ddim_config_path: Optional[Path] = None
        self.pnp_config_path: Optional[Path] = None

        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("AnyV2VRunner")

    def prepare_video(self, video_path: str) -> VideoInfo:
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")

        video_name = vp.stem
        self.logger.info(f"Preparing video: {video_name}")

        # Workspace in output_root since the source video folder is read-only
        work_dir = self.output_root / video_name / "anyv2v_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Frames should be directly in work_dir, not in a subdirectory
        # because AnyV2V expects demo/{video_name}/00000.png
        frames_dir = work_dir

        # Create symlink in demo folder so AnyV2V can find the work directory
        demo_link = self.demo_path / video_name
        if demo_link.exists() or demo_link.is_symlink():
            if demo_link.is_symlink():
                demo_link.unlink()
            elif demo_link.is_dir():
                shutil.rmtree(demo_link)
        demo_link.symlink_to(work_dir, target_is_directory=True)

        clip = VideoFileClip(str(vp))
        info = VideoInfo(
            video_name=video_name,
            video_path=str(vp),
            duration=clip.duration,
            fps=clip.fps,
            size=tuple(clip.size),
            work_dir=str(work_dir),
            frames_dir=str(frames_dir),
            demo_dir=str(work_dir)  # keep compatibility with older code
        )

        # Extract frames if they don't exist (save directly in work_dir, resized to 512x512)
        existing_frames = list(frames_dir.glob("*.png"))
        if not existing_frames:
            self.logger.info(f"Extracting frames from video to {frames_dir} (resizing to 512x512)")
            frame_count = 0
            import PIL.Image
            for i, frame in enumerate(clip.iter_frames()):
                # Convert to PIL Image and resize to 512x512
                pil_img = PIL.Image.fromarray(frame)
                pil_img_resized = pil_img.resize((512, 512), PIL.Image.LANCZOS)

                frame_path = frames_dir / f"{i:05d}.png"
                pil_img_resized.save(frame_path)
                frame_count += 1
            self.logger.info(f"Extracted and resized {frame_count} frames to 512x512")
        else:
            self.logger.info(f"Found {len(existing_frames)} existing frames, skipping extraction")

        clip.close()
        return info

    def generate_ddim_config(self, video_info: VideoInfo) -> Path:
        config_data = [{
            "active": True,
            "force_recompute_latents": True,
            "video_name": video_info.video_name,
            "recon_config": {"enable_recon": True}
        }]
        # out = self.i2vgen_xl_path / "configs" / "group_ddim_inversion" / "group_config.json"
        cfg_root = self.output_root / ".configs" / f"rank{self.rank}"
        out = cfg_root / "group_ddim_inversion.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        self.logger.info(f"Generated DDIM config: {out}")
        self.ddim_config_path = out
        return out

    def generate_pnp_config(self, video_info: VideoInfo, prompt: str, short_name: str) -> Path:
        """
        Generate a *single* Prompt‑and‑Paint entry that uses our combined two‑object colour change prompt.
        """
        edited_frame_dir = Path(video_info.demo_dir) / "color_change"
        edited_frame_dir.mkdir(parents=True, exist_ok=True)

        # Use short_name instead of the full prompt slug
        edited_first_frame = edited_frame_dir / f"{short_name}.png"

        cfg = EDITING_PARAMS
        config_data = [{
            "active": True,
            "task_name": "Prompt-Based-Editing",
            "video_name": video_info.video_name,
            "edited_first_frame_path": f"demo/{video_info.video_name}/color_change/{short_name}.png",
            "editing_prompt": prompt,
            "edited_video_name": short_name,
            "ddim_init_latents_t_idx": cfg["ddim_init_latents_t_idx"],
            "pnp_f_t": cfg["pnp_f_t"],
            "pnp_spatial_attn_t": cfg["pnp_spatial_attn_t"],
            "pnp_temp_attn_t": cfg["pnp_temp_attn_t"],
            "cfg": cfg["cfg"],
            "n_steps": cfg["n_steps"]
        }]

        # out = self.i2vgen_xl_path / "configs" / "group_pnp_edit" / "group_config.json"
        cfg_root = self.output_root / ".configs" / f"rank{self.rank}"
        out = cfg_root / "group_pnp_edit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        self.logger.info(f"Generated PnP config: {out}")
        self.pnp_config_path = out
        return out

    def edit_first_frame(self, video_info: VideoInfo, prompt: str, short_name: str) -> Path:
        """
        Produce the edited first frame using the repo's 'edit_image.py' helper.
        """
        edited_frame_dir = Path(video_info.demo_dir) / "color_change"
        edited_frame_dir.mkdir(parents=True, exist_ok=True)
        out_png = edited_frame_dir / f"{short_name}.png"

        if out_png.exists():
            self.logger.info(f"Edited first frame already exists: {out_png}")
            return out_png

        cmd = [
            sys.executable, "edit_image.py",
            "--video_path", video_info.video_path,
            "--input_dir", str(self.demo_path.parent),
            "--output_dir", str(edited_frame_dir),
            "--prompt", prompt,
            "--negative_prompt", NEGATIVE_PROMPT,
            "--seed", "42",
            "--output_filename", short_name
        ]
        try:
            self.logger.info(f"Editing first frame with prompt: {prompt}")
            _ = __import__("subprocess").run(
                cmd, cwd=self.anyv2v_root, capture_output=True, text=True, check=True
            )
            self.logger.info(f"Image editing completed: {out_png}")
            return out_png
        except __import__("subprocess").CalledProcessError as e:
            self.logger.error(f"Image editing failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
            raise

    def run_ddim_inversion(self, _video_info: VideoInfo) -> None:
        # If latents exist, skip
        # latents_dir = self.anyv2v_root / "inversions" / "i2vgen-xl" / _video_info.video_name / "ddim_latents"
        # if latents_dir.exists() and any(latents_dir.glob("*.pt")):
        #     self.logger.info("DDIM latents already exist; skipping inversion")
        #     return
        # Only skip if latents look COMPLETE; otherwise clean and recompute
        latents_dir = self.anyv2v_root / "inversions" / "i2vgen-xl" / _video_info.video_name / "ddim_latents"

        def _latents_complete(p):
            try:
                # Expect roughly 50 timesteps for the default scheduler; accept >=45 as "complete"
                return p.exists() and len(list(p.glob("*.pt"))) >= 45
            except Exception:
                return False

        if _latents_complete(latents_dir):
            self.logger.info("DDIM latents present and look complete; skipping inversion")
            return
        if latents_dir.exists():
            self.logger.warning(f"Incomplete/old latents found at {latents_dir}; removing to recompute")
            shutil.rmtree(latents_dir.parent, ignore_errors=True)  # remove whole video folder

        cmd = [
            sys.executable, "run_group_ddim_inversion.py",
            "--template_config", "configs/group_ddim_inversion/template.yaml",
            # "--configs_json", "configs/group_ddim_inversion/group_config.json"
            "--configs_json", str(self.ddim_config_path)
        ]
        try:
            self.logger.info("Running DDIM inversion ...")
            _ = __import__("subprocess").run(
                cmd, cwd=self.i2vgen_xl_path, capture_output=True, text=True, check=True
            )
            self.logger.info("DDIM inversion completed")
        except __import__("subprocess").CalledProcessError as e:
            self.logger.error(f"DDIM inversion failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
            raise

    def run_pnp_editing(self) -> None:
        cmd = [
            sys.executable, "run_group_pnp_edit.py",
            "--template_config", "configs/group_pnp_edit/template.yaml",
            # "--configs_json", "configs/group_pnp_edit/group_config.json"
            "--configs_json", str(self.pnp_config_path)
        ]
        try:
            self.logger.info("Running Prompt‑and‑Paint editing ...")
            _ = __import__("subprocess").run(
                cmd, cwd=self.i2vgen_xl_path, capture_output=True, text=True, check=True
            )
            self.logger.info("PnP editing completed")
        except __import__("subprocess").CalledProcessError as e:
            self.logger.error(f"PnP editing failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
            raise

    def collect_and_copy_result(self, video_info: VideoInfo, final_filename: str) -> Optional[Path]:
        """
        Look into AnyV2V results and copy the first found video to output_root/<video_name>/color_change/<final_filename>
        """
        results_dir = self.anyv2v_root / "Results" / "Prompt-Based-Editing" / "i2vgen-xl" / video_info.video_name
        out_dir = self.output_root / video_info.video_name / "color_change"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not results_dir.exists():
            self.logger.warning(f"No results dir found: {results_dir}")
            return None

        # Find an mp4 first; if none, accept gif
        candidates = list(results_dir.rglob("*.mp4"))
        if not candidates:
            candidates = list(results_dir.rglob("*.gif"))
        if not candidates:
            self.logger.warning(f"No generated video files found under: {results_dir}")
            return None

        src = candidates[0]
        dst = out_dir / final_filename
        shutil.copy2(src, dst)
        self.logger.info(f"Saved: {dst}")

        # Delete DDIM latents to save space
        latents_dir = self.anyv2v_root / "inversions" / "i2vgen-xl" / video_info.video_name
        if latents_dir.exists():
            shutil.rmtree(latents_dir, ignore_errors=True)
            self.logger.info(f"Removed DDIM latents to save space: {latents_dir}")

        # Remove png frames extracted in work_dir to save space
        frames = list(Path(video_info.frames_dir).glob("*.png"))
        for f in frames:
            try:
                f.unlink()
            except Exception:
                pass
        self.logger.info(f"Removed extracted frames to save space: {len(frames)} files")

        return dst


# ------------------------------ Main workflow from CSV ------------------------------

def select_names_for_run(all_names: List[str], output_root: Path, num: int, repeat: bool) -> List[str]:
    """
    Mimic the selection/filtering logic in the Diffusers script:
    - Skip names that already have a non‑empty exported file unless --repeat
    - Randomly sample up to num
    """
    import random

    done_names = set()
    # consider any file ending with .mp4 in the expected colourChange folder as 'done'
    for name in all_names:
        folder = output_root / name / "color_change"
        if folder.exists():
            for f in folder.glob("*.mp4"):
                if f.stat().st_size > 0:
                    done_names.add(name)
                    break

    todo = [n for n in all_names if repeat or n not in done_names]
    if len(todo) <= num:
        return todo
    return random.sample(todo, num)


def process_csv(
        csv_path: Path,
        output_root: Path,
        anyv2v_root: Path,
        num: int,
        repeat: bool,
) -> List[Path]:
    name_to_row = read_csv_like_diffusers(csv_path)
    if not name_to_row:
        logging.warning("No valid rows found in CSV.")
        return []

    # names = list(name_to_row.keys())
    # selected = select_names_for_run(names, output_root, num=num, repeat=repeat)
    #
    # logging.info(
    #     "CSV selection summary:\n"
    #     f"  total rows: {len(names)}\n"
    #     f"  selected for this run: {len(selected)}"
    # )
    names = sorted(list(name_to_row.keys()))  # deterministic order
    selected = select_names_for_run(names, output_root, num=num, repeat=repeat)

    # ----- torchrun-aware sharding -----
    rank, local_rank, world_size = get_dist_info()
    # each rank gets a slice; e.g. rank 0 -> 0, world_size, 2*world_size, ...
    selected_shard = selected[rank::world_size]
    logging.info(
        "Distributed selection summary:\n"
        f"  world_size: {world_size}  rank: {rank}  local_rank: {local_rank}\n"
        f"  total selected: {len(selected)}  this-rank: {len(selected_shard)}"
    )

    runner = AnyV2VRunner(anyv2v_root=anyv2v_root, output_root=output_root)

    saved_paths: List[Path] = []
    # for name in selected:
    for name in selected_shard:
        row = name_to_row[name]

        # Must have at least one object
        (obj1, col1), = assign_two_distinct_colours(row.objects)

        prompt = color_change_prompt(row.caption, obj1, col1)
        short_name = make_short_name(name, obj1, col1)

        # Prepare video and configs
        vinfo = runner.prepare_video(row.video_path)
        runner.generate_ddim_config(vinfo)
        runner.generate_pnp_config(vinfo, prompt, short_name)

        # First‑frame edit to seed the PnP
        edited_frame_path = runner.edit_first_frame(vinfo, prompt, short_name)

        if not edited_frame_path.exists():
            logging.error(f"Edited frame not generated: {edited_frame_path}")
            continue  # Skip this video and go to next

        # Run AnyV2V steps
        runner.run_ddim_inversion(vinfo)
        runner.run_pnp_editing()

        # Collect and rename
        final_name = make_final_filename(name, obj1, col1)
        saved = runner.collect_and_copy_result(vinfo, final_name)
        if saved is not None:
            saved_paths.append(saved)

    return saved_paths


# ------------------------------ CLI ------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnyV2V colour‑change pipeline (CSV‑driven)")
    p.add_argument("--csv", required=True, help="CSV path (same format your CosmosPredict2 script accepts)")
    p.add_argument("--output_root", required=True, help="Root path to store final outputs")
    p.add_argument("--anyv2v_root", required=True, help="Path to AnyV2V repository root")
    p.add_argument("-n", "--num", type=int, default=1, help="Number of videos to process (default: 1)")
    p.add_argument("--repeat", action="store_true", help="Re‑process even if outputs exist")
    return p


def main():
    rank, _, _ = get_dist_info()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - %(levelname)s - [{rank}] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler()
        ]
    )

    args = build_argparser().parse_args()
    csv_path = Path(args.csv).expanduser()
    output_root = Path(args.output_root).expanduser()
    anyv2v_root = Path(args.anyv2v_root).expanduser()

    output_root.mkdir(parents=True, exist_ok=True)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not anyv2v_root.exists():
        raise FileNotFoundError(f"AnyV2V root not found: {anyv2v_root}")

    # one GPU per rank when launched by torchrun
    restrict_visible_gpu_to_local_rank()

    # rank-aware stdout hygiene
    rank, _, _ = get_dist_info()
    saved = process_csv(csv_path, output_root, anyv2v_root, num=args.num, repeat=args.repeat)
    if rank == 0:
        print(f"Saved {len(saved)} video(s).")
        for p in saved:
            print(p)

    # saved = process_csv(csv_path, output_root, anyv2v_root, num=args.num, repeat=args.repeat)
    # print(f"Saved {len(saved)} video(s).")
    # for p in saved:
    #     print(p)


if __name__ == "__main__":
    main()
