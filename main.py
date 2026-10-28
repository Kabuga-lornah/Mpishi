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
PLAN_MODEL = os.getenv("MPISHI_PLAN_MODEL", "muse-wonder-1.0")   # used with web search when planning an item
PLAN_REASONING = os.getenv("MPISHI_PLAN_REASONING", "low")
MAX_ITEMS = 8
UNATTENDED_STEPS = [(300, "watch"), (600, "watch"), (1200, "alert"), (1800, "alert")]  # seconds with the stove on and nobody near
QUIET = {"state": "ok", "say": "", "timer_seconds": None, "goal_met": False}

SYSTEM_PROMPT = """You are Mpishi, a kitchen safety watcher. You see two photos from a fixed camera over a stove: PREVIOUS (older) and CURRENT.
The cook is making: {food}. Reply in {language}.
{goal_line}
{last_say_line}
Reply with ONLY a JSON object, no other text:
{{"pan_visible": true | false, "person_present": true | false, "burner_lit": true | false, "state": "ok" | "watch" | "alert" | "unsure", "say": "<text>", "timer_seconds": <integer or null>, "goal_met": true | false}}
Facts: pan_visible = a pan, pot or cooking appliance is in view. person_present = a person or hand is in view. burner_lit = a flame or glowing element is visible.
Rules for "say":
- Stay SILENT ("") by default. NEVER describe the cook, their hands, their movements or what they are doing. NEVER narrate.
- Speak only about the food or the stove, and only when the cook needs to act or know: boiling over, liquid almost gone, smoke, burning, a flame with no pot, something dangerous near the flame, or the goal being reached.
- At most one short sentence, then at most one action.
- Never claim texture, temperature, or doneness inside the food.
- "alert" only for smoke, flames outside the burner, boil-over, burning, or liquid nearly gone.
- If something looks unusual (for example possible packaging), say what you see and ask the cook to check. Do not order the cook to throw food away.
- Do not repeat your previous message.
- Set timer_seconds only if the cook asked for a timer; otherwise null.
- If the goal is met, "say" must say so in one short sentence.
- If you cannot tell, use "unsure" and "say" "".
"""

PLAN_PROMPT = """You are a careful cooking guide. The cook is putting this on the stove now: {food}. Their preference: {pref}. Reply in {language}.
If you have a web search tool, use it to check typical times and visual cues for cooking this on a home stovetop. Home cooking is slower and less even than a restaurant kitchen, so give ranges.
Reply with ONLY a JSON object: {{"intro": "<one or two short sentences>", "checkpoints": [{{"at_minutes": <integer>, "say": "<one short reminder>"}}]}}
Rules:
- "intro": a rough total time as a range, the main stages the cook can SEE (for example pale, golden, dark), and any assumption you made. If the method matters and no preference was given (for example steamed or boiled), assume the most common one and name the alternative. Say the times are estimates.
- "checkpoints": at most 6, in time order, counted in minutes from now, at least 3 minutes apart, none later than 240. Each says what to look for or do then: check the liquid level, stir, lower the heat, turn the food, check doneness BY HAND or with a food thermometer, or that it will burn if left longer.
- Never state a safe internal temperature.
- If the dish takes under 3 minutes, return an empty checkpoints list."""

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
    preference: str = Field(default="", max_length=200)
    language: str = "en"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)


class ItemRequest(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    preference: str = Field(default="", max_length=200)


def _tri(v):
    return v if isinstance(v, bool) else None


def parse_reply(text) -> dict:
    """Turn the model's text into a safe result. Never raises."""
    unsure = {"state": "unsure", "say": "", "timer_seconds": None, "goal_met": False,
              "pan_visible": None, "person_present": None, "burner_lit": None}
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
    return {"state": state, "say": say.strip()[:300], "timer_seconds": timer, "goal_met": data.get("goal_met") is True,
            "pan_visible": _tri(data.get("pan_visible")), "person_present": _tri(data.get("person_present")),
            "burner_lit": _tri(data.get("burner_lit"))}


def parse_plan(text) -> dict:
    """Validate a cooking plan from the model. Never raises."""
    empty = {"intro": "", "checkpoints": []}
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return empty
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return empty
    points = []
    for c in data.get("checkpoints") or []:
        if not isinstance(c, dict):
            continue
        at, say = c.get("at_minutes"), c.get("say")
        if isinstance(at, bool) or not isinstance(at, int) or not 1 <= at <= 240 or not isinstance(say, str) or not say.strip():
            continue
        points.append({"at_minutes": at, "say": say.strip()[:200]})
    points.sort(key=lambda p: p["at_minutes"])
    intro = data.get("intro") if isinstance(data.get("intro"), str) else ""
    return {"intro": intro.strip()[:300], "checkpoints": points[:6]}


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


def _chat(content, shrunk=()):
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": content}], temperature=0.2, max_tokens=2000)
    client = get_client()
    try:
        try:
            resp = client.chat.completions.create(reasoning_effort=REASONING, **kwargs)
        except (openai.BadRequestError, TypeError):
            resp = client.chat.completions.create(**kwargs)  # reasoning_effort not accepted
    except Exception as e:
        log.error("Meta call failed: %s status=%s images=%s detail=%s", type(e).__name__,
                  getattr(e, "status_code", None), describe(list(shrunk)), str(e)[:400])
        raise
    return resp.choices[0].message.content


def call_model(prompt: str, images: list) -> dict:
    shrunk = [shrink_jpeg(i) for i in images]
    content = [{"type": "text", "text": prompt}] + [to_data_url(i) for i in shrunk]
    return parse_reply(_chat(content, shrunk))


def _output_text(resp) -> str:
    """Pull the text out of a Responses API result."""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts = []
    for item in getattr(resp, "output", None) or []:
        for c in getattr(item, "content", None) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _grounded_text(prompt: str) -> str:
    """Ask with web search turned on (Responses API + web_search tool)."""
    client = get_client().with_options(timeout=90)
    resp = client.responses.create(model=PLAN_MODEL, input=prompt, max_output_tokens=4000,
                                   reasoning={"effort": PLAN_REASONING},
                                   tools=[{"type": "web_search", "search_context_size": "medium"}])
    return _output_text(resp)


def make_plan(food: str, preference: str, language: str) -> dict:
    """Plan one item. Tries web search first, then falls back to the model's own knowledge."""
    pref = preference or "none given (assume the most common method and name the alternative)"
    prompt = PLAN_PROMPT.format(food=food, pref=pref, language=language)
    try:
        plan = parse_plan(_grounded_text(prompt))
        if plan["intro"] or plan["checkpoints"]:
            return {**plan, "source": "web"}
    except Exception as e:
        log.error("Grounded plan failed: %s %s", type(e).__name__, str(e)[:300])
    return {**parse_plan(_chat(prompt)), "source": "memory"}


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


MESSAGES = {
    "no_pan": {"English": "I cannot see the pan. Please aim the camera at the stove.",
               "Swahili": "Siwezi kuona sufuria. Elekeza kamera kwenye jiko."},
    "away": {"English": "The stove is still on and nobody has been near it for {m} minutes.",
             "Swahili": "Jiko bado linawaka na hakuna mtu karibu kwa dakika {m}."},
}


def msg(kind: str, language: str, minutes: int = 0) -> str:
    return MESSAGES[kind].get(language, MESSAGES[kind]["English"]).format(m=minutes)


def due_messages(s: dict, now: float) -> list:
    """Messages driven by the clock and the last known facts, not by what the model said."""
    out = []
    for item in s["items"]:
        elapsed, points = now - item["started"], item["checkpoints"]
        while item["plan_idx"] < len(points) and elapsed >= points[item["plan_idx"]]["at_minutes"] * 60:
            out.append({"state": "watch", "say": f'{item["name"]}: {points[item["plan_idx"]]["say"]}'})
            item["plan_idx"] += 1
    f = s["facts"]
    if f.get("burner_lit") is True and f.get("person_present") is False and f.get("pan_visible") is not False:
        if s["away_since"] is None:
            s["away_since"] = now
        away = now - s["away_since"]
        while s["away_level"] < len(UNATTENDED_STEPS) and away >= UNATTENDED_STEPS[s["away_level"]][0]:
            secs, state = UNATTENDED_STEPS[s["away_level"]]
            out.append({"state": state, "say": msg("away", s["language"], secs // 60)})
            s["away_level"] += 1
    else:
        s["away_since"], s["away_level"] = None, 0
    return out


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


def new_item(name: str, preference: str, language: str, now: float) -> dict:
    try:
        plan = make_plan(name, preference, language)
    except Exception as e:
        log.error("Plan failed: %s", type(e).__name__)
        plan = {"intro": "", "checkpoints": [], "source": "none"}
    return {"name": name, "started": now, "plan_idx": 0, **plan}


def public(item: dict) -> dict:
    return {k: item[k] for k in ("name", "intro", "checkpoints", "source")}


@app.post("/api/session")
def start(req: StartRequest):
    now = time.time()
    for old in [k for k, v in SESSIONS.items() if now - v["created"] > SESSION_TTL_SECONDS]:
        SESSIONS.pop(old, None)
    language = LANGUAGES.get(req.language, "English")
    items = [new_item(req.food.strip(), req.preference.strip(), language, now)] if req.food.strip() else []
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"food": req.food.strip() or "whatever is in the pan (say what you see)",
                     "goal": req.goal.strip(), "language": language,
                     "prev": None, "last": None, "last_call": 0.0, "calls": 0, "created": now, "last_say": "",
                     "items": items, "facts": {}, "away_since": None, "away_level": 0, "no_pan": 0}
    return {"session_id": sid, "items": [public(i) for i in items]}


@app.post("/api/session/{sid}/items")
def add_item(sid: str, req: ItemRequest):
    s = get_session(sid)
    if len(s["items"]) >= MAX_ITEMS:
        raise HTTPException(409, f"At most {MAX_ITEMS} items per cook.")
    item = new_item(req.name.strip(), req.preference.strip(), s["language"], time.time())
    s["items"].append(item)
    s["food"] = ", ".join(i["name"] for i in s["items"])
    return public(item)


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
        return {"status": "baseline", **QUIET, "notices": due_messages(s, now)}
    diff = frame_changed(prev, data)
    if diff < CHANGE_THRESHOLD and now - s["last_call"] < HEARTBEAT_SECONDS:
        return {"status": "unchanged", **QUIET, "diff": round(diff, 1), "notices": due_messages(s, now)}
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
    for k in ("pan_visible", "person_present", "burner_lit"):
        if result[k] is not None:
            s["facts"][k] = result[k]
    notices = []
    if result["pan_visible"] is False:
        s["no_pan"] += 1
        if s["no_pan"] == 2:
            notices.append({"state": "watch", "say": msg("no_pan", s["language"])})
    elif result["pan_visible"] is True:
        s["no_pan"] = 0
    notices += due_messages(s, now)
    return {"status": "analysed", "diff": round(diff, 1), **result, "notices": notices}


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