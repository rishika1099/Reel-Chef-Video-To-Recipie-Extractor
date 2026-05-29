# 🍳 ReelChef

**Turn any cooking video into a structured recipe.**

ReelChef watches a cooking video frame by frame, describes what is happening,
groups those descriptions into step-like segments, and asks an LLM to write a
clean recipe: title, ingredients, numbered steps with timestamps, key moments,
and a short summary. Upload a clip, get a recipe you can actually follow.

---

## Table of contents

- [How it works](#how-it-works)
- [Demo flow](#demo-flow)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Local setup](#local-setup)
- [Configuration and secrets](#configuration-and-secrets)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Optional: URL based Gradio app](#optional-url-based-gradio-app)
- [Tuning the pipeline](#tuning-the-pipeline)
- [Limitations](#limitations)

---

## How it works

ReelChef runs a four-stage vision-to-language pipeline:

```
  Video (.mp4 / .mov / .m4v)
        |
        v
  1. Extract frames        ffmpeg samples the video at N fps  ->  frames/*.jpg
        |
        v
  2. Caption frames        BLIP (image-to-text) describes each frame on CPU
        |                  ->  outputs/captions.json  [{time_s, frame, caption}]
        v
  3. Build timeline        consecutive near-duplicate captions are merged into
        |                  step-like segments via string similarity
        |                  ->  outputs/timeline.json  [{start_s, end_s, caption}]
        v
  4. Summarize (OpenAI)    an LLM turns the timeline into a Markdown recipe
                           ->  outputs/recipe.md
```

Each stage writes a downloadable artifact, so you can inspect exactly what the
model saw at every step.

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `src/extract_frames.py` | Runs `ffmpeg` to sample frames at a chosen fps. |
| 2 | `src/caption_frames.py` | Captions frames with `Salesforce/blip-image-captioning-base` (CPU). |
| 3 | `src/build_timeline.py` | Merges similar consecutive captions using `difflib.SequenceMatcher`. |
| 4 | `src/summarize_with_openai.py` | Sends the timeline to OpenAI and writes a Markdown recipe. |

---

## Demo flow

1. Open the app.
2. Upload a short cooking video.
3. (Optional) Adjust sampling rate, max frames, similarity threshold, and model in the sidebar.
4. Click **Cook the recipe**.
5. Watch the four stages run, read the generated recipe, and download any artifact.

---

## Tech stack

- **Streamlit** for the web UI (primary app).
- **ffmpeg** for frame extraction.
- **PyTorch + Hugging Face Transformers (BLIP)** for image captioning, CPU only.
- **Pillow** for image loading.
- **OpenAI** (`gpt-4o-mini` / `gpt-4o`) for recipe generation via the Responses API.
- **Gradio + yt-dlp** for an optional URL-based variant.

---

## Project structure

```
ReelChef/
├── app_streamlit.py            # Main Streamlit app (upload -> recipe)
├── app_gradio.py               # Optional Gradio app (URL -> recipe)
├── src/
│   ├── extract_frames.py       # Stage 1: ffmpeg frame sampling
│   ├── caption_frames.py       # Stage 2: BLIP captioning
│   ├── build_timeline.py       # Stage 3: timeline segmentation
│   └── summarize_with_openai.py# Stage 4: OpenAI recipe generation
├── requirements.txt            # Python deps for the Streamlit app
├── requirements-gradio.txt     # Extra deps for the Gradio app
├── packages.txt                # System deps (ffmpeg) for Streamlit Cloud
├── .streamlit/
│   └── secrets.toml.example    # Template for the OpenAI key
├── .env.example                # Template for local env vars
└── .gitignore
```

---

## Local setup

Requires Python 3.10+ and `ffmpeg` installed on your system.

```bash
# 1. Clone
git clone https://github.com/<your-username>/ReelChef.git
cd ReelChef

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg if you don't have it
#    macOS:   brew install ffmpeg
#    Ubuntu:  sudo apt-get install ffmpeg

# 5. Add your OpenAI key (see next section)
cp .env.example .env             # then edit .env

# 6. Run
streamlit run app_streamlit.py
```

The first run downloads the BLIP model weights (a few hundred MB), so give it a
moment.

---

## Configuration and secrets

ReelChef needs an OpenAI API key. **Never commit your key.** The key is resolved
lazily at call time from, in order:

1. The `OPENAI_API_KEY` environment variable.
2. A local `.env` file (loaded automatically via `python-dotenv`).
3. Streamlit secrets (`.streamlit/secrets.toml` locally, or the Secrets panel on Streamlit Cloud).

`.env` and `.streamlit/secrets.toml` are both gitignored. Use the provided
`.env.example` and `.streamlit/secrets.toml.example` as templates.

The app loads fine without a key; it just shows a warning and skips recipe
generation until a key is provided.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and select your `ReelChef` repo.
4. Set **Main file path** to `app_streamlit.py`.
5. Open **Advanced settings -> Secrets** and paste:

   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

6. Click **Deploy**.

`packages.txt` tells Streamlit Cloud to install `ffmpeg` automatically, and
`requirements.txt` covers the Python side.

---

## Optional: URL based Gradio app

`app_gradio.py` accepts a video URL instead of an upload and downloads it with
`yt-dlp`. Run it locally:

```bash
pip install -r requirements.txt -r requirements-gradio.txt
python app_gradio.py
```

> Note: downloading from third-party video sites can be unreliable and may be
> against some platforms' terms of service. The upload-based Streamlit app is
> the recommended path.

---

## Tuning the pipeline

All exposed in the Streamlit sidebar:

| Setting | Default | Effect |
|---------|---------|--------|
| Frame sampling rate (fps) | 1 | Higher = more frames, more detail, slower. |
| Max frames | 300 | Caps total frames captioned to keep CPU runs reasonable. |
| Segmentation threshold | 0.72 | Higher = stricter, more (shorter) segments. |
| OpenAI model | gpt-4o-mini | `gpt-4o` is stronger but pricier. |

---

## Limitations

- Captioning runs on CPU, so long videos are slow. Keep clips short or lower the max-frame cap.
- BLIP captions are approximate; the recipe is the model's best inference, not ground truth.
- The free Streamlit Cloud tier has limited memory; very large videos or high frame counts may exceed it.
- Audio and on-screen text are not transcribed; the pipeline is purely visual.
