import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


@torch.inference_mode()
def caption_frames(frames_dir: str, out_json: str, fps: int = 1, max_frames: int = 300) -> None:
    """Caption each sampled frame with BLIP and save a list of {time_s, frame, caption}."""
    device = "cpu"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    frames = sorted(Path(frames_dir).glob("frame_*.jpg"))[:max_frames]
    results = []

    for i, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)

        results.append({"time_s": round(i / fps, 2), "frame": fp.name, "caption": caption})

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
