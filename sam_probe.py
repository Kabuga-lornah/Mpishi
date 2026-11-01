"""Probe Meta's SAM 3.1 so we can see the exact request and response format before wiring it into Mpishi.

Usage (PowerShell, in the project folder):
    $env:MODEL_API_KEY = "your-key"
    .venv\\Scripts\\python sam_probe.py frame.png "pot"

Works with an image (.png/.jpg) or a short video (.mp4). The key is read from the environment and never printed.
"""
import base64
import json
import mimetypes
import os
import re
import sys
import time

BASE = "https://api.meta.ai/v1"
VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v"}


def upload(path: str, key: str) -> str:
    import httpx
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        r = httpx.post(f"{BASE}/files", headers={"Authorization": f"Bearer {key}"}, data={"purpose": "user_data"},
                       files={"file": (os.path.basename(path), f, mime)}, timeout=120)
    r.raise_for_status()
    return r.json()["id"]


def ask(prompt: str, media_part: dict, key: str):
    import httpx
    payload = {"model": "sam-3.1", "stream": True, "metadata": {"mask_encoding": "lossless", "include_confidence": "true"},
               "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, media_part]}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    lines, t0 = [], time.time()
    with httpx.stream("POST", f"{BASE}/responses", headers=headers, json=payload, timeout=180) as r:
        if r.status_code != 200:
            return r.status_code, r.read().decode("utf-8", "replace")[:500], [], time.time() - t0
        for raw in r.iter_lines():
            if raw.startswith("data: "):
                if raw[6:] == "[DONE]":
                    break
                lines.append(raw[6:])
    return 200, "", lines, time.time() - t0


def summarise(lines: list):
    """Count event types and find the box records, whether or not the JSON escaped the special tokens."""
    types, decoded = {}, []
    for line in lines:
        try:
            obj = json.loads(line)
            types[obj.get("type", "?") if isinstance(obj, dict) else "?"] = types.get(obj.get("type", "?") if isinstance(obj, dict) else "?", 0) + 1
            decoded.append(json.dumps(obj, ensure_ascii=False))
        except ValueError:
            types["(not json)"] = types.get("(not json)", 0) + 1
            decoded.append(line)
    return types, re.findall(r"<\|box;[^|]*\|>", "\n".join(decoded))


def main() -> None:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if len(sys.argv) < 3 or not key:
        sys.exit(__doc__ + "\nMissing arguments or MODEL_API_KEY.")
    path, prompt = sys.argv[1], sys.argv[2]
    if os.path.splitext(path)[1].lower() in VIDEO_EXT:
        variants = [("video by file_id", lambda: {"type": "input_video", "file_id": upload(path, key)})]
    else:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        url = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
        variants = [("image as image_url data URL", lambda: {"type": "input_image", "image_url": url}),
                    ("image by file_id", lambda: {"type": "input_image", "file_id": upload(path, key)})]
    for name, build in variants:
        print(f"\n=== {name} | prompt: {prompt!r} ===")
        try:
            status, err, lines, secs = ask(prompt, build(), key)
        except Exception as e:
            print("failed:", type(e).__name__, str(e)[:200])
            continue
        print(f"HTTP {status} in {secs:.1f}s, {len(lines)} stream events")
        if status != 200:
            print(err)
            continue
        types, boxes = summarise(lines)
        print("event types:", types)
        print("boxes found:", len(boxes))
        for b in boxes[:5]:
            print("   ", b[:200])
        for line in lines[:3] + lines[-2:]:
            print("sample:", re.sub(r"data=[^|]{60,}", "data=<masks omitted>", line)[:300])
        out = "sam_probe_" + os.path.splitext(os.path.basename(path))[0] + ".txt"
        open(out, "w", encoding="utf-8").write("\n".join(lines))
        print("full raw output saved to", out, "(masks are large: do not paste that file)")
        break      # this request shape was accepted


if __name__ == "__main__":
    main()