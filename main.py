"""Mpishi - an AI kitchen safety watcher.

Camera -> Meta vision model -> structured facts -> safety/timing engine -> voice/UI alerts.
The cook does NOT have to tell Mpishi when they leave the kitchen; unattended detection is automatic.
"""
import base64
import io
import asyncio
import contextvars
import difflib
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
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
from pydantic import BaseModel, Field

try:
    import websockets
except ImportError:  # voice input needs: pip install websockets
    websockets = None

BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
MODEL = os.getenv("MPISHI_MODEL", "muse-spark-1.3")
PLAN_MODEL = os.getenv("MPISHI_PLAN_MODEL", "muse-wonder-1.0")
REASONING = os.getenv("MPISHI_REASONING", "minimal")
PLAN_REASONING = os.getenv("MPISHI_PLAN_REASONING", "low")
CHANGE_THRESHOLD = float(os.getenv("MPISHI_CHANGE_THRESHOLD", "4.0"))
HEARTBEAT_SECONDS = int(os.getenv("MPISHI_HEARTBEAT_SECONDS", "45"))
MAX_FRAME_BYTES = 3_000_000
VALID_STATES = {"ok", "watch", "alert", "unsure"}
LANGUAGES = {"en": "English", "sw": "Kiswahili"}
ACCESS_CODE = os.getenv("MPISHI_ACCESS_CODE", "").strip()
MAX_CALLS_PER_SESSION = int(os.getenv("MPISHI_MAX_CALLS", "400"))
SESSION_TTL_SECONDS = 4 * 3600
MAX_ITEMS = 8
# Real-world escalation: stove on and nobody near for 5, 10, 20 and 30 minutes. Demo mode divides these by DEMO_SPEED.
UNATTENDED_STEPS = [(300, "watch"), (600, "watch"), (1200, "alert"), (1800, "alert")]
DEMO_SPEED = float(os.getenv("MPISHI_DEMO_SPEED", "20"))
QUIET = {
    "state": "ok",
    "say": "",
    "timer_seconds": None,
    "goal_met": False,
    "pan_visible": None,
    "person_present": None,
    "burner_lit": None,
}

VOICE_URL = os.getenv("MPISHI_VOICE_URL", "wss://api.meta.ai/v1/asr/realtime")
VOICE_MODEL = "muse-voice-transcribe-1.0"
VOICE_MAX_SECONDS = 20
VOICE_USD_PER_HOUR = 0.18
# USD per million tokens (input, output), from the Models page. Cached-input discounts are ignored, so costs are an upper bound.
PRICES = {"muse-spark-1.3": (1.25, 4.25), "muse-spark-1.2": (1.25, 4.25), "muse-spark-1.1": (1.25, 4.25), "muse-wonder-1.0": (4.0, 20.0)}
USAGE = contextvars.ContextVar("usage", default=None)
log = logging.getLogger("uvicorn.error")
app = FastAPI(title="Mpishi")
SESSIONS: dict[str, dict] = {}
_client = None

SYSTEM_PROMPT = """You are Mpishi, a kitchen safety watcher. You see PREVIOUS (older) and CURRENT photos from a fixed camera over a stove.
The cook is making: {food}. Reply in {language}.
{goal_line}
{last_say_line}
Reply with ONLY a JSON object:
{{"pan_visible": true | false, "person_present": true | false, "burner_lit": true | false, "state": "ok" | "watch" | "alert" | "unsure", "say": "<text>", "timer_seconds": <integer or null>, "goal_met": true | false}}

Definitions:
- pan_visible = a pan, pot or cooking appliance is clearly visible.
- person_present = a person OR a visible hand/arm is currently in the camera's useful cooking area.
- burner_lit = a visible flame or visibly glowing heating element. If you cannot tell, use null.

Rules:
- Stay SILENT ("") by default. NEVER describe the cook, their hands, their movements or what they are doing. NEVER narrate.
- Speak only when the cook needs to know or do something about the food/stove.
- Alert examples: visible smoke, visible burning, boil-over, flame outside the burner, or a clearly dangerous visible condition.
- Do NOT claim internal temperature, internal doneness, or facts the camera cannot verify.
- If you cannot tell, use state="unsure" and say="".
- Never repeat the previous message unless the situation has materially changed.
- Set timer_seconds only if the cook explicitly asked for a timer.
- Set goal_met true only when the photos clearly support the goal.
- Keep say to one short sentence plus at most one action.
- Output must be valid JSON and nothing else.
"""

PLAN_PROMPT = """You are Mpishi, a careful cooking guide. The cook is putting this on the stove now: {food}. Preference: {pref}. Reply in {language}.
If a web search tool is available, use it to check typical home-stovetop times and visible cues. Give estimates, not guarantees.
Reply ONLY with JSON:
{{"intro":"<short estimate>","checkpoints":[{{"at_minutes":<integer>,"say":"<short reminder>"}}]}}
Rules:
- Intro: an approximate time range, then only the stages visible from above (steam, bubbles, colour), never what is inside the food. Say the times are estimates. If the method matters and no preference was given, assume the most common one and name the alternative.
- Checkpoints: short reminders (under 15 words), counted in minutes from now, to look, stir, turn, lower the heat or check the liquid level. Prefer checks that do not spoil the food.
- Never state a safe internal temperature.
- Never claim a food is definitely safe or cooked internally based on appearance.
- At most 6 checkpoints, 3 minutes apart, none later than 240 minutes.
- If the dish takes under 3 minutes, return no checkpoints.
"""

ASK_PROMPT = """You are Mpishi, a kitchen watcher. You see the CURRENT photo from a camera over a stove.
The cook is making: {food}. Reply in {language}.
Question: "{question}"
Reply ONLY with JSON:
{{"state":"ok"|"watch"|"alert"|"unsure","say":"<at most two short sentences>","timer_seconds":<integer or null>}}
For questions about what is in the pan, use the photo. For general cooking questions, answer briefly from general knowledge and call it an estimate. Never claim internal temperature or internal doneness. If unsure, say so and suggest checking by hand.
"""

# User-facing messages are centralized so English/Kiswahili can be changed consistently.
MESSAGES = {
    "en": {
        "no_pan": "I cannot see the pan clearly. Please check the camera view.",
        "quiet": "Okay. I will stay quiet and keep watching.",
        "voice_fail": "I could not hear that. Tap the microphone to try again.",
        "timer_set": "Timer set for {m}.",
        "away_300": "The stove appears unattended. Please check the kitchen.",
        "away_600": "The stove is still on and nobody appears to be near it. Please check the stove.",
        "away_1200": "Please check the stove now. It appears to have been unattended for a while.",
        "away_1800": "Urgent: the stove still appears unattended. Please check it now.",
        "watching": "Mpishi is watching. All quiet.",
        "watching_away": "Mpishi is watching the stove automatically.",
        "start": "Mpishi is watching. Point the camera at the cooking area.",
        "connection": "I lost the connection. Please check the stove yourself.",
        "timer_done": "Timer done. Check the food.",
        "uncertain": "I am not sure. Please check the food yourself.",
        "camera_secure": "The camera needs a secure page. Use localhost or an HTTPS link.",
        "camera_blocked": "Camera access was blocked. Allow camera access and try again.",
        "planning": "Planning your cook...",
        "start_button": "Start watching",
        "stop": "Stop",
        "ask": "Ask",
        "add": "Add",
        "planning_short": "Planning...",
        "what_cooking": "What are you cooking? (optional)",
        "goal": "Tell me when... (optional)",
        "preference": "How do you like it? (optional)",
        "language": "Language",
        "demo": "Demo mode: safety intervals accelerated",
        "camera_hint": "An extra set of eyes on your stove. Not a smoke alarm.",
        "add_item": "Add another item, e.g. steamed cabbage",
        "ask_placeholder": "Ask: is it ready yet?",
        "guide": "Cooking guide (estimates; check by hand)",
        "last_check": "Last check",
        "not_watching": "MPISHI IS NOT WATCHING",
        "check_yourself": "Please check the stove yourself.",
        "from_search": "from web search",
        "from_memory": "from model knowledge",
        "no_guide": "no guide",
        "after": "after",
        "minutes": "min",
        "seconds": "seconds",
        "minutes_word": "minutes",
    },
    "sw": {
        "no_pan": "Siwezi kuona sufuria vizuri. Tafadhali angalia mwonekano wa kamera.",
        "quiet": "Sawa. Nitakaa kimya na kuendelea kuangalia.",
        "voice_fail": "Sikusikia vizuri. Bonyeza kipaza sauti ujaribu tena.",
        "timer_set": "Kipima muda kimewekwa: {m}.",
        "away_300": "Inaonekana jiko halina mtu anayelichunga. Tafadhali angalia jikoni.",
        "away_600": "Jiko bado linawaka na hakuna mtu anayeonekana kuwa karibu. Tafadhali angalia jiko.",
        "away_1200": "Tafadhali angalia jiko sasa. Inaonekana limeachwa bila mtu kwa muda.",
        "away_1800": "Tahadhari: jiko bado linaonekana kuachwa bila mtu. Tafadhali angalia sasa.",
        "watching": "Mpishi anaangalia. Kila kitu kiko sawa.",
        "watching_away": "Mpishi anaendelea kulinda jiko kiotomatiki.",
        "start": "Mpishi anaangalia. Elekeza kamera kwenye sehemu ya kupikia.",
        "connection": "Nimepoteza muunganisho. Tafadhali angalia jiko mwenyewe.",
        "timer_done": "Muda umekwisha. Angalia chakula.",
        "uncertain": "Sina uhakika. Tafadhali angalia chakula mwenyewe.",
        "camera_secure": "Kamera inahitaji ukurasa salama. Tumia localhost au kiungo cha HTTPS.",
        "camera_blocked": "Kamera imezuiwa. Ruhusu matumizi ya kamera kisha ujaribu tena.",
        "planning": "Mpishi anapanga upishi wako...",
        "start_button": "Anza kuangalia",
        "stop": "Simamisha",
        "ask": "Uliza",
        "add": "Ongeza",
        "planning_short": "Inapanga...",
        "what_cooking": "Unapika nini? (hiari)",
        "goal": "Niambie wakati... (hiari)",
        "preference": "Unapenda chakula kiweje? (hiari)",
        "language": "Lugha",
        "demo": "Hali ya onyesho: vipindi vya tahadhari vimeharakishwa",
        "camera_hint": "Macho ya ziada ya kuangalia jiko lako. Si king'ora cha moshi.",
        "add_item": "Ongeza chakula kingine, kwa mfano kabeji ya mvuke",
        "ask_placeholder": "Uliza: kimeiva tayari?",
        "guide": "Mwongozo wa upishi (makadirio; hakikisha mwenyewe)",
        "last_check": "Ukaguzi wa mwisho",
        "not_watching": "MPISHI HAANGALII",
        "check_yourself": "Tafadhali angalia jiko mwenyewe.",
        "from_search": "kutoka utafutaji wa wavuti",
        "from_memory": "kutoka maarifa ya modeli",
        "no_guide": "hakuna mwongozo",
        "after": "baada ya",
        "minutes": "dak",
        "seconds": "sekunde",
        "minutes_word": "dakika",
    },
}


def t(lang_code: str, key: str) -> str:
    return MESSAGES.get(lang_code, MESSAGES["en"]).get(key, MESSAGES["en"].get(key, key))


def new_usage() -> dict:
    return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0, "voice_s": 0.0}


def _add_usage(resp, model: str) -> None:
    """Record tokens and cost of one model call against the session that triggered it."""
    u, usage = USAGE.get(), getattr(resp, "usage", None)
    if u is None or usage is None:
        return
    tin = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
    tout = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
    p_in, p_out = PRICES.get(model, PRICES["muse-spark-1.3"])
    u["calls"] += 1
    u["tokens_in"] += tin
    u["tokens_out"] += tout
    u["usd"] += (tin * p_in + tout * p_out) / 1_000_000


def usage_view(u: dict) -> dict:
    return {"calls": u["calls"], "tokens_in": u["tokens_in"], "tokens_out": u["tokens_out"],
            "voice_s": round(u["voice_s"], 1), "usd": round(u["usd"] + u["voice_s"] / 3600 * VOICE_USD_PER_HOUR, 4)}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = (os.getenv("MODEL_API_KEY") or "").strip().strip('"\'')
        if not key:
            raise HTTPException(500, "MODEL_API_KEY is not set. Copy .env.example to .env and add your Meta model key.")
        _client = OpenAI(base_url=BASE_URL, api_key=key, timeout=40)
    return _client


class StartRequest(BaseModel):
    food: str = Field(default="", max_length=200)
    goal: str = Field(default="", max_length=200)
    preference: str = Field(default="", max_length=200)
    language: str = "en"
    demo: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)


class ItemRequest(BaseModel):
    name: str = Field(min_length=2, max_length=200)
    preference: str = Field(default="", max_length=200)


def _tri(v):
    return v if isinstance(v, bool) else None


def parse_reply(text) -> dict:
    unsure = {**QUIET, "state": "unsure"}
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
    return {
        "state": state,
        "say": say.strip()[:300],
        "timer_seconds": timer,
        "goal_met": data.get("goal_met") is True,
        "pan_visible": _tri(data.get("pan_visible")),
        "person_present": _tri(data.get("person_present")),
        "burner_lit": _tri(data.get("burner_lit")),
    }


def parse_plan(text) -> dict:
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
    return {"intro": (data.get("intro") if isinstance(data.get("intro"), str) else "").strip()[:300], "checkpoints": points[:6]}


def frame_changed(prev: bytes, cur: bytes) -> float:
    def small(b):
        return Image.open(io.BytesIO(b)).convert("L").resize((64, 64))
    return ImageStat.Stat(ImageChops.difference(small(prev), small(cur))).mean[0]


def to_data_url(jpeg: bytes) -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def shrink_jpeg(data: bytes, max_side: int = 768, quality: int = 75) -> bytes:
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


def _chat(content, shrunk=(), model=None):
    kwargs = dict(model=model or MODEL, messages=[{"role": "user", "content": content}], temperature=0.2, max_tokens=2000)
    client = get_client()
    try:
        try:
            resp = client.chat.completions.create(reasoning_effort=REASONING, **kwargs)
        except (openai.BadRequestError, TypeError):
            resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        log.error("Meta call failed: %s status=%s images=%s detail=%s", type(e).__name__, getattr(e, "status_code", None), describe(list(shrunk)), str(e)[:400])
        raise
    _add_usage(resp, model or MODEL)
    return resp.choices[0].message.content or ""


def call_model(prompt: str, images: list) -> dict:
    shrunk = [shrink_jpeg(i) for i in images]
    content = [{"type": "text", "text": prompt}] + [to_data_url(i) for i in shrunk]
    return parse_reply(_chat(content, shrunk))


def _output_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text
    parts = []
    for item in getattr(resp, "output", None) or []:
        for c in getattr(item, "content", None) or []:
            value = getattr(c, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _grounded_text(prompt: str) -> str:
    client = get_client().with_options(timeout=90)
    resp = client.responses.create(
        model=PLAN_MODEL,
        input=prompt,
        max_output_tokens=4000,
        reasoning={"effort": PLAN_REASONING},
        tools=[{"type": "web_search", "search_context_size": "medium"}],
    )
    _add_usage(resp, PLAN_MODEL)
    return _output_text(resp)


def make_plan(food: str, preference: str, language: str) -> dict:
    pref = preference or "none given; assume a common stovetop method and state the assumption"
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


def human_time(secs: float, lang_code: str) -> str:
    secs = max(0, int(round(secs)))
    if secs < 60:
        return f"{secs} {t(lang_code, 'seconds')}"
    return f"{secs // 60} {t(lang_code, 'minutes_word')}"


def due_messages(s: dict, now: float) -> list:
    """Drive reminders from the clock and last known vision facts, not from manual user input."""
    out = []
    speed = s["speed"]

    # Food-specific cooking checkpoints.
    for item in s["items"]:
        elapsed = (now - item["started"]) * speed
        points = item["checkpoints"]
        while item["plan_idx"] < len(points) and elapsed >= points[item["plan_idx"]]["at_minutes"] * 60:
            point = points[item["plan_idx"]]
            out.append({"state": "watch", "say": f'{item["name"]}: {point["say"]}'})
            item["plan_idx"] += 1

    # Automatic unattended-stove detection. No button is required.
    f = s["facts"]
    unattended = (
        f.get("burner_lit") is True
        and f.get("person_present") is False
        and f.get("pan_visible") is not False
    )

    if unattended:
        if s["away_since"] is None:
            s["away_since"] = now
            s["away_level"] = 0
        away = now - s["away_since"]
        while s["away_level"] < len(UNATTENDED_STEPS):
            real_seconds, state = UNATTENDED_STEPS[s["away_level"]]
            limit = real_seconds / speed
            if away < limit:
                break
            key = f"away_{real_seconds}"
            out.append({"state": state, "say": t(s["language_code"], key)})
            s["away_level"] += 1
    else:
        s["away_since"] = None
        s["away_level"] = 0

    return out


@app.middleware("http")
async def check_access(request: Request, call_next):
    """If MPISHI_ACCESS_CODE is set, every /api/ call except /api/health needs the X-Access-Code header.
    Leave it unset for local use; set it before sharing a tunnel or deploying."""
    path = request.url.path
    if ACCESS_CODE and path.startswith("/api/") and path != "/api/health":
        supplied = request.headers.get("x-access-code", "")
        if not secrets.compare_digest(supplied.encode(), ACCESS_CODE.encode()):
            return JSONResponse({"detail": "Wrong or missing access code."}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "protected": bool(ACCESS_CODE), "voice": websockets is not None, "key_set": bool(os.getenv("MODEL_API_KEY"))}


@app.get("/api/selftest")
def selftest():
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
    usage = new_usage()
    USAGE.set(usage)
    lang_code = req.language if req.language in LANGUAGES else "en"
    language = LANGUAGES[lang_code]
    items = [new_item(req.food.strip(), req.preference.strip(), language, now)] if req.food.strip() else []
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {
        "food": req.food.strip() or "whatever is in the cooking area",
        "goal": req.goal.strip(),
        "language": language,
        "language_code": lang_code,
        "prev": None,
        "last": None,
        "last_call": 0.0,
        "calls": 0,
        "created": now,
        "last_say": "",
        "items": items,
        "facts": {},
        "away_since": None,
        "away_level": 0,
        "no_pan": 0,
        "speed": DEMO_SPEED if req.demo else 1.0,
        "usage": usage,
    }
    return {"session_id": sid, "items": [public(i) for i in items], "language": lang_code}


@app.post("/api/session/{sid}/items")
def add_item(sid: str, req: ItemRequest):
    s = get_session(sid)
    USAGE.set(s["usage"])
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

    prev = s["last"]
    now = time.time()
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

    goal_line = (
        f'The cook wants to be told when: {s["goal"]}. Set "goal_met" to true only when the photos clearly show this.'
        if s["goal"] else 'There is no goal. Always set "goal_met" to false.'
    )
    last = s["last_say"].replace('"', "'")
    last_say_line = f'Your previous message was: "{last}".' if last else "You have not spoken yet."
    prompt = SYSTEM_PROMPT.format(food=s["food"], language=s["language"], goal_line=goal_line, last_say_line=last_say_line)

    try:
        USAGE.set(s["usage"])
        result = await run_in_threadpool(call_model, prompt, [prev, data])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())

    if result["pan_visible"] is False and result["state"] != "alert":
        result["say"] = ""      # the server announces "cannot see the pan" once; no repeated camera complaints
    if result["say"]:
        s["last_say"] = result["say"]
    s["prev"], s["last_call"] = prev, now

    clears = s["away_since"] is not None and (result["person_present"] is True or result["burner_lit"] is False)
    s["clear_votes"] = s.get("clear_votes", 0) + 1 if clears else 0
    for key in ("pan_visible", "person_present", "burner_lit"):
        if result[key] is None or (clears and s["clear_votes"] < 2 and key != "pan_visible"):
            continue                # one odd reading does not cancel the stove countdown
        s["facts"][key] = result[key]

    notices = []
    if result["pan_visible"] is False:
        s["no_pan"] += 1
        if s["no_pan"] == 2:
            notices.append({"state": "watch", "say": t(s["language_code"], "no_pan")})
    elif result["pan_visible"] is True:
        s["no_pan"] = 0

    notices.extend(due_messages(s, now))
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


NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60}
TIMER_RE = re.compile(r"\b(?:set (?:a )?timer for|timer for|remind me in|wake me in)\s+([\w-]+)\s*(second|sec|minute|min|hour)s?\b")
ADD_RE = re.compile(r"^(?:please\s+)?(?:(?:i(?:'m| am)\s+)?adding|add|i added|(?:i(?:'m| am)\s+)?putting in)\s+(.{2,80}?)[.!?\s]*$")
NO_RE = re.compile(r"^(?:no|nope|nothing|not now|no thanks|no thank you|i(?:'m| am) (?:fine|good)|all good)\b")


def parse_timer(text: str):
    m = TIMER_RE.search((text or "").lower())
    if not m:
        return None
    word, unit = m.groups()
    n = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
    if not n:
        return None
    secs = n * (3600 if unit == "hour" else 60 if unit.startswith("min") else 1)
    return secs if 1 <= secs <= 7200 else None


def handle_utterance(s: dict, text: str) -> dict:
    """Route what the cook said: decline, set a timer, add an item, or answer a question about the pan or cooking."""
    text, lc = (text or "").strip(), s["language_code"]
    base = {"heard": text, "state": "ok", "say": "", "timer_seconds": None, "item": None}
    if not text:
        return {**base, "kind": "empty", "say": t(lc, "voice_fail")}
    low = text.lower()
    if NO_RE.match(low):
        return {**base, "kind": "decline", "say": t(lc, "quiet")}
    secs = parse_timer(low)
    if secs:
        label = f"{secs // 60} min" if secs >= 60 and secs % 60 == 0 else f"{secs} sec"
        return {**base, "kind": "timer", "timer_seconds": secs, "say": t(lc, "timer_set").format(m=label)}
    added = ADD_RE.match(low)
    if added and len(s["items"]) < MAX_ITEMS:
        item = new_item(added.group(1), "", s["language"], time.time())
        s["items"].append(item)
        s["food"] = ", ".join(i["name"] for i in s["items"])
        return {**base, "kind": "add_item", "say": item["intro"], "item": public(item)}
    if not s["last"]:
        return {**base, "kind": "empty", "say": t(lc, "voice_fail")}
    r = call_model(ASK_PROMPT.format(food=s["food"], language=s["language"], question=text.replace('"', "'")), [s["last"]])
    return {**base, "kind": "answer", "state": r["state"], "say": r["say"], "timer_seconds": r["timer_seconds"]}


class UtteranceRequest(BaseModel):
    text: str = Field(default="", max_length=300)


@app.post("/api/session/{sid}/utterance")
def utterance(sid: str, req: UtteranceRequest):
    s = get_session(sid)
    USAGE.set(s["usage"])
    try:
        return handle_utterance(s, req.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())


@app.get("/api/session/{sid}/usage")
def usage(sid: str):
    return usage_view(get_session(sid)["usage"])


def _voice_handshake(lang) -> dict:
    h = {"authorization": {"accessToken": f"Bearer {get_client().api_key}"}, "audioEncoding": "PCM_16KHZ", "model": VOICE_MODEL,
         "mode": "ENDPOINTING", "partialMode": "CUMULATIVE", "emitAudioProgress": False, "keywords": ["Mpishi"]}
    if lang != "sw":      # Swahili is not among Meta's listed languages: leave detection automatic
        h["languageBias"] = ["English"]
    return h


def strip_wake(text: str):
    """Hands-free: did the sentence start with the wake word 'Mpishi'? Returns (heard, rest of the sentence)."""
    m = re.match(r"^\W*(?:(?:hey|hi|hello|ok|okay)\W+)?([\w']+)\W*(.*)$", (text or "").strip(), re.S)
    if not m:
        return False, ""
    first = m.group(1).lower()
    ok = first in ("mpishi", "mpisha", "pishi", "mpeshi", "mpichi") or difflib.SequenceMatcher(None, first, "mpishi").ratio() >= 0.72
    return (True, m.group(2).strip()) if ok else (False, "")


@app.websocket("/ws/voice/{sid}")
async def voice(ws: WebSocket, sid: str):
    """Relay the browser's microphone (PCM16, 16 kHz, mono) to Meta's realtime transcription, then act on what was said."""
    s = SESSIONS.get(sid)
    code_ok = not ACCESS_CODE or secrets.compare_digest(ws.query_params.get("code", "").encode(), ACCESS_CODE.encode())
    if not s or not code_ok or websockets is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    USAGE.set(s["usage"])
    handshake = _voice_handshake(ws.query_params.get("lang"))
    sent, tasks = 0, []
    try:
        async with websockets.connect(f"{VOICE_URL}?sessionId=stream-{uuid.uuid4()}", open_timeout=10) as up:
            await up.send(json.dumps(handshake))
            ack = json.loads(await asyncio.wait_for(up.recv(), 10))
            if "sessionId" not in ack:
                await ws.send_json({"type": "error", "message": str(ack.get("message", ack))[:200]})
                return
            await ws.send_json({"type": "ready"})

            async def pump_up():
                nonlocal sent
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes"):
                        sent += len(msg["bytes"])
                        await up.send(msg["bytes"])
                    elif msg.get("text") == "end":
                        await up.send(json.dumps({"type": "endStream"}))
                        return

            async def pump_down():
                async for raw in up:
                    if isinstance(raw, bytes):
                        continue
                    data = json.loads(raw)
                    kind = data.get("type")
                    if kind == "transcript" and data.get("transcript"):
                        await ws.send_json({"type": "partial", "text": data["transcript"]})
                    elif kind == "speechComplete":
                        return data.get("transcript", "")
                    elif kind == "error":
                        raise RuntimeError(str(data.get("message", data))[:200])
                return ""

            tasks = [asyncio.create_task(pump_up()), asyncio.create_task(pump_down())]
            text = await asyncio.wait_for(tasks[1], VOICE_MAX_SECONDS)
        s["usage"]["voice_s"] += sent / 32000
        result = await run_in_threadpool(handle_utterance, s, text)
        await ws.send_json({"type": "final", **result})
    except Exception as e:
        log.error("Voice failed: %s %s", type(e).__name__, str(e)[:200])
        try:
            await ws.send_json({"type": "error", "message": type(e).__name__})
        except Exception:
            pass
    finally:
        for task in tasks:
            task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws/listen/{sid}")
async def listen(ws: WebSocket, sid: str):
    """Hands-free: keep Meta's realtime transcription open; act only on sentences that start with 'Mpishi'
    (or that follow a 'window' from the page, e.g. right after a check-in question)."""
    s = SESSIONS.get(sid)
    code_ok = not ACCESS_CODE or secrets.compare_digest(ws.query_params.get("code", "").encode(), ACCESS_CODE.encode())
    if not s or not code_ok or websockets is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    USAGE.set(s["usage"])
    awake_until, tasks = 0.0, []
    try:
        async with websockets.connect(f"{VOICE_URL}?sessionId=stream-{uuid.uuid4()}", open_timeout=10) as up:
            await up.send(json.dumps(_voice_handshake(ws.query_params.get("lang"))))
            ack = json.loads(await asyncio.wait_for(up.recv(), 10))
            if "sessionId" not in ack:
                await ws.send_json({"type": "error", "message": str(ack.get("message", ack))[:200]})
                return
            await ws.send_json({"type": "ready"})

            async def pump_up():
                nonlocal awake_until
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("bytes"):
                        s["usage"]["voice_s"] += len(msg["bytes"]) / 32000
                        await up.send(msg["bytes"])
                    elif msg.get("text") == "window":
                        awake_until = time.time() + 10

            async def pump_down():
                nonlocal awake_until
                async for raw in up:
                    if isinstance(raw, bytes):
                        continue
                    data = json.loads(raw)
                    if data.get("type") == "error":
                        raise RuntimeError(str(data.get("message", data))[:200])
                    if data.get("type") != "speechComplete" or not (data.get("transcript") or "").strip():
                        continue
                    heard, rest = strip_wake(data["transcript"])
                    if heard:
                        command = rest
                    elif time.time() < awake_until:
                        command = data["transcript"].strip()
                    else:
                        continue                      # ordinary kitchen talk is ignored
                    if not command:                   # just "Mpishi": listen for the next sentence
                        awake_until = time.time() + 10
                        await ws.send_json({"type": "awake"})
                        continue
                    awake_until = 0.0
                    result = await run_in_threadpool(handle_utterance, s, command)
                    await ws.send_json({"type": "final", **result})

            tasks = [asyncio.create_task(pump_up()), asyncio.create_task(pump_down())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                d.result()
    except Exception as e:
        log.error("Hands-free failed: %s %s", type(e).__name__, str(e)[:200])
        try:
            await ws.send_json({"type": "error", "message": type(e).__name__})
        except Exception:
            pass
    finally:
        for task in tasks:
            task.cancel()
        try:
            await ws.close()
        except Exception:
            pass


@app.delete("/api/session/{sid}")
def stop(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")