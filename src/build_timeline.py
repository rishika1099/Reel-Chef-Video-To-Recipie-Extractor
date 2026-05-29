import json
from difflib import SequenceMatcher
from pathlib import Path


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_timeline(captions_json: str, out_json: str, thresh: float = 0.72) -> None:
    """Merge consecutive near-duplicate captions into step-like timeline segments."""
    data = json.loads(Path(captions_json).read_text(encoding="utf-8"))
    if not data:
        raise ValueError("No captions found.")

    segments = []
    cur = {"start_s": data[0]["time_s"], "end_s": data[0]["time_s"], "caption": data[0]["caption"]}

    for item in data[1:]:
        if _sim(cur["caption"], item["caption"]) >= thresh:
            cur["end_s"] = item["time_s"]
        else:
            segments.append(cur)
            cur = {"start_s": item["time_s"], "end_s": item["time_s"], "caption": item["caption"]}

    segments.append(cur)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(segments, indent=2), encoding="utf-8")
