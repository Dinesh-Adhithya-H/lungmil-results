"""
Create MP4 slideshow videos from individual patient showcase figures.

For each task, combines the 4 individual patient PNGs (split=0) into a
16:9 letterboxed MP4 (1920×1080) with each patient shown for 5 seconds.

Output: figures/video/trajectories_{task}_showcase.mp4

Usage: sbatch analysis/submit_make_showcase_videos.sh
"""

import subprocess
import tempfile
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
UMAP_DIR = ROOT / "figures" / "interpretability" / "patient_rep_umap"
OUT_DIR  = ROOT / "figures" / "video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
SPLIT = 0
DURATION_PER_PAT = 5   # seconds each patient shown
FPS = 25
W, H = 1920, 1080
BG = "0xFAF6F2"        # warm background colour (hex for ffmpeg)

# Locate bundled ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    sys.path.insert(0, "/home/aih/dinesh.haridoss/.local/lib/python3.12/site-packages")
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

print(f"ffmpeg: {FFMPEG}", flush=True)


def make_video(task: str):
    pat_dir = UMAP_DIR / task / f"split{SPLIT}"
    pngs = [pat_dir / f"trajectories_{task}_showcase_pat{i}.png" for i in range(1, 5)]

    missing = [p for p in pngs if not p.exists()]
    if missing:
        print(f"  [skip] missing files: {missing}", flush=True)
        return

    out_mp4 = OUT_DIR / f"trajectories_{task}_showcase.mp4"

    # Write concat list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for png in pngs:
            f.write(f"file '{png}'\n")
            f.write(f"duration {DURATION_PER_PAT}\n")
        # ffmpeg concat demuxer needs final image repeated with 0 duration
        f.write(f"file '{pngs[-1]}'\n")
        f.write("duration 0.04\n")
        list_path = f.name

    # ffmpeg command: concat → scale+pad to 1920x1080 letterbox → h264 mp4
    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vf", (
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG},"
            f"fps={FPS}"
        ),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    print(f"  Running ffmpeg...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error]\n{result.stderr[-500:]}", flush=True)
    else:
        size_mb = out_mp4.stat().st_size / 1e6
        print(f"  Saved {out_mp4.name}  ({size_mb:.1f} MB)", flush=True)

    Path(list_path).unlink(missing_ok=True)


def main():
    for task in TASKS:
        print(f"\n=== {task} ===", flush=True)
        make_video(task)
    print(f"\nDone. → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
