import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

from src.extract_frames import extract_frames
from src.caption_frames import caption_frames
from src.build_timeline import build_timeline
from src.summarize_with_openai import summarize_timeline


def download_video(url: str) -> str:
    """Download a video from a URL with yt-dlp into a temp file."""
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "video.mp4")
    subprocess.run(["yt-dlp", "-f", "mp4", "-o", out_path, url], check=True)
    return out_path


def run_pipeline(video_url, fps=1, max_frames=300, thresh=0.72, model="gpt-4o-mini"):
    if not video_url:
        return "Please provide a video URL.", None, None, None

    shutil.rmtree("frames", ignore_errors=True)
    shutil.rmtree("outputs", ignore_errors=True)
    Path("frames").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    try:
        video_path = download_video(video_url)

        extract_frames(video_path, "frames", fps=int(fps))
        caption_frames("frames", "outputs/captions.json", fps=int(fps), max_frames=int(max_frames))
        build_timeline("outputs/captions.json", "outputs/timeline.json", thresh=float(thresh))
        md = summarize_timeline("outputs/timeline.json", "outputs/recipe.md", model=model)

        return md, "outputs/captions.json", "outputs/timeline.json", "outputs/recipe.md"

    except Exception as e:
        return f"Error: {str(e)}", None, None, None


with gr.Blocks(title="ReelChef") as demo:
    gr.Markdown("# 🍳 ReelChef")
    gr.Markdown(
        "Paste a YouTube or video URL. ReelChef turns the video into a structured "
        "recipe using a vision-to-language pipeline."
    )

    video_url = gr.Textbox(label="Video URL", placeholder="https://www.youtube.com/watch?v=...")

    with gr.Row():
        fps = gr.Slider(1, 2, value=1, step=1, label="FPS (frame sampling)")
        max_frames = gr.Slider(100, 800, value=300, step=50, label="Max frames")
        thresh = gr.Slider(0.6, 0.9, value=0.72, step=0.01, label="Segmentation threshold")

    model = gr.Dropdown(["gpt-4o-mini", "gpt-4o"], value="gpt-4o-mini", label="OpenAI model")

    run_btn = gr.Button("Cook the recipe")

    output_md = gr.Markdown(label="Generated Recipe")
    captions_file = gr.File(label="captions.json")
    timeline_file = gr.File(label="timeline.json")
    recipe_file = gr.File(label="recipe.md")

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_url, fps, max_frames, thresh, model],
        outputs=[output_md, captions_file, timeline_file, recipe_file],
    )

if __name__ == "__main__":
    demo.launch()
