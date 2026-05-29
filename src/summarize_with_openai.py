import os
import json
from pathlib import Path

from openai import OpenAI

# Load a local .env for development if python-dotenv is available. This is a
# safe no-op when the package or file is missing. On Streamlit Community Cloud
# the key is supplied through st.secrets instead (see _get_api_key).
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:
    pass


def _get_api_key() -> str:
    """Resolve the OpenAI key lazily from the environment or Streamlit secrets.

    Resolving at call time (rather than import time) keeps the app importable
    without a key, so the UI can load and show a friendly message instead of
    crashing on startup.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st

            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment, a local "
            ".env file, or .streamlit/secrets.toml."
        )
    return key


def summarize_timeline(
    timeline_json: str,
    out_md: str,
    model: str = "gpt-4o-mini",
) -> str:
    client = OpenAI(api_key=_get_api_key())

    timeline = json.loads(Path(timeline_json).read_text(encoding="utf-8"))

    prompt = f"""
You are a cooking assistant.

Given a timeline of video segments with timestamps and visual descriptions:
1) Create a recipe title.
2) List ingredients (explicit + reasonable inferred).
3) Write numbered cooking steps in order with timestamp ranges.
4) Identify 3-6 key moments and why they matter.
5) List missing/unclear steps and how to improve the pipeline.
6) Write a 150-250 word summary.

Return Markdown with headings.

Timeline JSON:
{json.dumps(timeline, indent=2)}
""".strip()

    resp = client.responses.create(model=model, input=prompt)
    text = resp.output_text

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(text, encoding="utf-8")
    return text
