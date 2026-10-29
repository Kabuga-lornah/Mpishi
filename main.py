"""Mpishi - an AI kitchen safety watcher.

Camera -> Meta vision model -> structured facts -> safety/timing engine -> voice/UI alerts.
The cook does NOT have to tell Mpishi when they leave the kitchen; unattended detection is automatic.
"""
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
Only describe what is visible. Never claim internal temperature or internal doneness. If unsure, say so.
"""

# User-facing messages are centralized so English/Kiswahili can be changed consistently.
MESSAGES = {
    "en": {
        "no_pan": "I cannot see the pan clearly. Please check the camera view.",
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
    return {"ok": True, "model": MODEL, "protected": bool(ACCESS_CODE), "key_set": bool(os.getenv("MODEL_API_KEY"))}


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
    }
    return {"session_id": sid, "items": [public(i) for i in items], "language": lang_code}


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
        result = await run_in_threadpool(call_model, prompt, [prev, data])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())

    if result["say"]:
        s["last_say"] = result["say"]
    s["prev"], s["last_call"] = prev, now

    for key in ("pan_visible", "person_present", "burner_lit"):
        if result[key] is not None:
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


@app.delete("/api/session/{sid}")
def stop(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")