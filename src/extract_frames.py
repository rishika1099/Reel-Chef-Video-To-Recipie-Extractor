import subprocess
from pathlib import Path


def extract_frames(video_path: str, out_dir: str, fps: int = 1) -> None:
    """Sample the video into JPG frames at `fps` frames per second using ffmpeg."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", f"{out_dir}/frame_%05d.jpg"],
        check=True,
    )
