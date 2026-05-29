import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from src.extract_frames import extract_frames
from src.caption_frames import caption_frames
from src.build_timeline import build_timeline
from src.summarize_with_openai import summarize_timeline

st.set_page_config(page_title="ReelChef", page_icon="🍳", layout="wide")
st.title("🍳 ReelChef")
st.caption("Turn any cooking video into a structured recipe.")


def _has_key() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        return False


def download_video(url: str) -> str:
    """Download a video from a URL with yt-dlp into a temp file."""
    tmp_dir = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "video.mp4")
    subprocess.run(["yt-dlp", "-f", "mp4/best", "-o", out_path, url], check=True)
    return out_path


with st.sidebar:
    st.header("Settings")
    fps = st.slider("Frame sampling rate (fps)", 1, 2, 1)
    max_frames = st.number_input("Max frames (CPU friendly)", 50, 1200, 300, 50)
    thresh = st.slider("Segmentation similarity threshold", 0.60, 0.90, 0.72, 0.01)
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"])

if not _has_key():
    st.warning(
        "No OpenAI API key found. Add `OPENAI_API_KEY` to your environment, a "
        "local `.env` file, or Streamlit secrets to generate recipes."
    )

tab_upload, tab_url = st.tabs(["Upload a file", "Paste a link"])
with tab_upload:
    uploaded = st.file_uploader("Upload a cooking video", type=["mp4", "mov", "m4v"])
    if uploaded:
        st.video(uploaded)
with tab_url:
    video_url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    st.caption(
        "Works for most public video URLs. Some sites (e.g. YouTube, Instagram) "
        "may block downloads from cloud servers; if a link fails, upload the file instead."
    )

run = st.button("Cook the recipe", type="primary")

if run:
    if not uploaded and not video_url:
        st.error("Upload a video or paste a link first.")
        st.stop()

    shutil.rmtree("frames", ignore_errors=True)
    shutil.rmtree("outputs", ignore_errors=True)
    Path("frames").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.getbuffer())
            video_path = tmp.name
    else:
        try:
            with st.spinner("Downloading video from URL..."):
                video_path = download_video(video_url)
        except Exception as e:
            st.error(
                "Could not download that link. The site may block server-side "
                "downloads; try uploading the file instead."
            )
            st.exception(e)
            st.stop()

    try:
        st.subheader("1) Extracting frames")
        with st.spinner("Running ffmpeg..."):
            extract_frames(video_path, "frames", fps=int(fps))
        st.success("Frames extracted.")

        st.subheader("2) Captioning frames (BLIP on CPU)")
        with st.spinner("Captioning frames..."):
            caption_frames("frames", "outputs/captions.json", fps=int(fps), max_frames=int(max_frames))
        st.success("Captions saved to outputs/captions.json")

        st.subheader("3) Building timeline")
        with st.spinner("Grouping into step-like segments..."):
            build_timeline("outputs/captions.json", "outputs/timeline.json", thresh=float(thresh))
        st.success("Timeline saved to outputs/timeline.json")

        st.subheader("4) Generating recipe (OpenAI)")
        with st.spinner("Calling OpenAI..."):
            md = summarize_timeline("outputs/timeline.json", "outputs/recipe.md", model=model)
        st.success("Recipe saved to outputs/recipe.md")

        st.markdown("## Recipe Report")
        st.markdown(md)

        st.markdown("## Downloads")
        st.download_button("Download captions.json", Path("outputs/captions.json").read_bytes(), "captions.json", "application/json")
        st.download_button("Download timeline.json", Path("outputs/timeline.json").read_bytes(), "timeline.json", "application/json")
        st.download_button("Download recipe.md", Path("outputs/recipe.md").read_bytes(), "recipe.md", "text/markdown")

    except Exception as e:
        st.exception(e)
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
