"""Mpishi: a kitchen watcher. Phone camera -> Muse Spark -> short spoken alerts."""
import base64
import io
import json
import logging
import os
import re
import secrets
import time
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import openai
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
from pydantic import BaseModel, Field

BASE_URL = "https://api.meta.ai/v1"
MODEL = os.getenv("MPISHI_MODEL", "muse-spark-1.3")
REASONING = os.getenv("MPISHI_REASONING", "minimal")
CHANGE_THRESHOLD = 4.0      # mean pixel difference (0-255) that counts as "changed"
HEARTBEAT_SECONDS = 45      # always re-check at least this often, even if nothing moved
MAX_FRAME_BYTES = 3_000_000
VALID_STATES = {"ok", "watch", "alert", "unsure"}
LANGUAGES = {"en": "English", "sw": "Swahili"}
ACCESS_CODE = os.getenv("MPISHI_ACCESS_CODE", "")
MAX_CALLS_PER_SESSION = int(os.getenv("MPISHI_MAX_CALLS", "400"))
SESSION_TTL_SECONDS = 4 * 3600

SYSTEM_PROMPT = """You are Mpishi, a kitchen watcher. You see two photos from a fixed camera over a stove: PREVIOUS (older) and CURRENT.
The cook is making: {food}. Reply in {language}.
{goal_line}
{last_say_line}
Reply with ONLY a JSON object, no other text:
{{"state": "ok" | "watch" | "alert" | "unsure", "say": "<text>", "timer_seconds": <integer or null>, "goal_met": true | false}}
Rules:
- "say" is at most two short sentences: what you see, then one action. Use "" if nothing noteworthy changed.
- Describe only what is visible: colour, bubbling, smoke, steam, boiling over, bursting, items moving or missing.
- Never claim texture, temperature, or doneness inside the food.
- "alert" only for smoke, flames outside the burner, boil-over, burning, or hot oil with no food that is smoking.
- If something looks unusual (for example possible packaging), say what you see and ask the cook to check. Do not order the cook to throw food away.
- Set timer_seconds only if the cook asked for a timer or you advise a specific extra cooking time; otherwise null.
- If the goal is met, "say" must say so in one short sentence.
- Do not repeat your previous message. If nothing has changed since then, set "say" to "".
- If no pan, pot, cooking appliance or food is visible, say once, in one short sentence, that you cannot see the pan. If your previous message already said that, set "say" to "" and state to "unsure".
- If you cannot tell, use "unsure"."""

ASK_PROMPT = """You are Mpishi, a kitchen watcher. You see one CURRENT photo from a camera over a stove.
The cook is making: {food}. Reply in {language}.
The cook asks: "{question}"
Reply with ONLY a JSON object: {{"state": "ok" | "watch" | "alert" | "unsure", "say": "<at most two short sentences>", "timer_seconds": <integer or null>}}
Describe only what is visible. Never claim texture or inside temperature. If unsure, say so and suggest checking by hand."""

log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Mpishi")
SESSIONS: dict = {}
_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = (os.getenv("MODEL_API_KEY") or "").strip().strip("\"'")
        if not key:
            raise HTTPException(500, "MODEL_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=40)
    return _client


class StartRequest(BaseModel):
    food: str = Field(default="", max_length=200)
    goal: str = Field(default="", max_length=200)
    language: str = "en"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)


def parse_reply(text) -> dict:
    """Turn the model's text into a safe result. Never raises."""
    unsure = {"state": "unsure", "say": "", "timer_seconds": None, "goal_met": False}
    if not text:
        return unsure
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return unsure
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return unsure
    state = data.get("state") if data.get("state") in VALID_STATES else "unsure"
    say = data.get("say") if isinstance(data.get("say"), str) else ""
    timer = data.get("timer_seconds")
    if isinstance(timer, bool) or not isinstance(timer, int) or not 1 <= timer <= 7200:
        timer = None
    return {"state": state, "say": say.strip()[:300], "timer_seconds": timer, "goal_met": data.get("goal_met") is True}


def frame_changed(prev: bytes, cur: bytes) -> float:
    """Mean grey-level difference between two JPEGs, 0 (same) to 255."""
    def small(b):
        return Image.open(io.BytesIO(b)).convert("L").resize((64, 64))
    return ImageStat.Stat(ImageChops.difference(small(prev), small(cur))).mean[0]


def to_data_url(jpeg: bytes) -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def shrink_jpeg(data: bytes, max_side: int = 768, quality: int = 75) -> bytes:
    """Re-encode a JPEG so its longest side is at most max_side (keeps phone portrait frames small)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality)
    return out.getvalue()


def describe(jpegs: list) -> list:
    out = []
    for b in jpegs:
        try:
            out.append(f"{Image.open(io.BytesIO(b)).size}/{len(b) // 1024}KB")
        except Exception:
            out.append("unreadable")
    return out


def call_model(prompt: str, images: list) -> dict:
    shrunk = [shrink_jpeg(i) for i in images]
    content = [{"type": "text", "text": prompt}] + [to_data_url(i) for i in shrunk]
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": content}],
                  temperature=0.2, max_tokens=2000)
    client = get_client()
    try:
        try:
            resp = client.chat.completions.create(reasoning_effort=REASONING, **kwargs)
        except (openai.BadRequestError, TypeError):
            resp = client.chat.completions.create(**kwargs)  # reasoning_effort not accepted
    except Exception as e:
        log.error("Meta call failed: %s status=%s images=%s detail=%s", type(e).__name__,
                  getattr(e, "status_code", None), describe(shrunk), str(e)[:400])
        raise
    return parse_reply(resp.choices[0].message.content)


def valid_jpeg(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def get_session(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Session not found. Start a new session.")
    return s


@app.middleware("http")
async def check_access(request: Request, call_next):
    path = request.url.path
    if ACCESS_CODE and path.startswith("/api/") and path != "/api/health":
        supplied = request.headers.get("x-access-code", "")
        if not secrets.compare_digest(supplied.encode(), ACCESS_CODE.encode()):
            return JSONResponse({"detail": "Wrong or missing access code."}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "protected": bool(ACCESS_CODE), "key_set": bool(os.getenv("MODEL_API_KEY"))}


@app.get("/api/selftest")
def selftest():
    """Ask Meta a tiny question from this server and report the exact error class (needs the access code header)."""
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": "Reply with the single word ok."}], max_tokens=300)
    try:
        client = get_client()
        try:
            r = client.chat.completions.create(reasoning_effort=REASONING, **kwargs)
        except (openai.BadRequestError, TypeError):
            r = client.chat.completions.create(**kwargs)
        return {"ok": True, "reply": (r.choices[0].message.content or "")[:40]}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "status": getattr(e, "status_code", None)}


@app.post("/api/session")
def start(req: StartRequest):
    now = time.time()
    for old in [k for k, v in SESSIONS.items() if now - v["created"] > SESSION_TTL_SECONDS]:
        SESSIONS.pop(old, None)
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"food": req.food.strip() or "whatever is in the pan (say what you see)",
                     "goal": req.goal.strip(), "language": LANGUAGES.get(req.language, "English"),
                     "prev": None, "last": None, "last_call": 0.0, "calls": 0, "created": now, "last_say": ""}
    return {"session_id": sid}


@app.post("/api/session/{sid}/frame")
async def frame(sid: str, file: UploadFile = File(...)):
    s = get_session(sid)
    data = await file.read()
    if len(data) > MAX_FRAME_BYTES or not valid_jpeg(data):
        raise HTTPException(400, "Frame must be a JPEG under 3 MB.")
    prev, now = s["last"], time.time()
    s["last"] = data
    if prev is None:
        s["prev"] = data
        s["last_call"] = now
        return {"status": "baseline", "state": "ok", "say": "", "timer_seconds": None}
    diff = frame_changed(prev, data)
    if diff < CHANGE_THRESHOLD and now - s["last_call"] < HEARTBEAT_SECONDS:
        return {"status": "unchanged", "state": "ok", "say": "", "timer_seconds": None, "diff": round(diff, 1)}
    s["calls"] += 1
    if s["calls"] > MAX_CALLS_PER_SESSION:
        raise HTTPException(429, "Session limit reached. Stop and start again.")
    goal_line = (f'The cook wants to be told when: {s["goal"]}. Set "goal_met" to true only when the photos clearly show this.'
                 if s["goal"] else 'There is no goal. Always set "goal_met" to false.')
    last = s["last_say"].replace('"', "'")
    last_say_line = f'Your previous message to the cook was: "{last}".' if last else "You have not said anything yet."
    prompt = SYSTEM_PROMPT.format(food=s["food"], language=s["language"], goal_line=goal_line, last_say_line=last_say_line)
    try:
        result = await run_in_threadpool(call_model, prompt, [prev, data])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())
    if result["say"]:
        s["last_say"] = result["say"]
    s["prev"], s["last_call"] = prev, now
    return {"status": "analysed", "diff": round(diff, 1), **result}


@app.post("/api/session/{sid}/ask")
def ask(sid: str, req: AskRequest):
    s = get_session(sid)
    if not s["last"]:
        raise HTTPException(409, "No camera frame yet. Wait a few seconds and ask again.")
    prompt = ASK_PROMPT.format(food=s["food"], language=s["language"], question=req.question.replace('"', "'"))
    try:
        return {"status": "answered", **call_model(prompt, [s["last"]])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())


@app.delete("/api/session/{sid}")
def stop(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")