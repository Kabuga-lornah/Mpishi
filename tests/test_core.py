import io
from types import SimpleNamespace
from PIL import Image
import main as app_main
from main import (parse_reply, parse_plan, frame_changed, due_messages, _output_text, make_plan,
                  MESSAGES, UNATTENDED_STEPS, SYSTEM_PROMPT, PLAN_PROMPT,
                  parse_timer, handle_utterance, new_usage, _add_usage, USAGE, usage_view, strip_wake)


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), color).save(buf, "JPEG")
    return buf.getvalue()


def item(name, points, started=0.0):
    return {"name": name, "started": started, "plan_idx": 0, "intro": "", "checkpoints": list(points), "source": "web"}


def sess(items=(), facts=None, speed=1.0, lang="en"):
    return {"items": list(items), "facts": facts or {}, "away_since": None, "away_level": 0,
            "language": "Kiswahili" if lang == "sw" else "English", "language_code": lang, "speed": speed}

AWAY = {"burner_lit": True, "person_present": False, "pan_visible": True}


def test_parse_good():
    r = parse_reply('```json\n{"state":"alert","say":"Smoke. Turn it off.","timer_seconds":300,"person_present":false,"burner_lit":true}\n```')
    assert r["state"] == "alert" and r["say"] == "Smoke. Turn it off." and r["timer_seconds"] == 300
    assert r["person_present"] is False and r["burner_lit"] is True and r["pan_visible"] is None and r["goal_met"] is False


def test_parse_garbage_is_safe():
    assert parse_reply("not json")["state"] == "unsure" and parse_reply(None)["say"] == ""
    r = parse_reply('{"state":"boom","say":5,"timer_seconds":-4,"burner_lit":"yes"}')
    assert r["state"] == "unsure" and r["say"] == "" and r["timer_seconds"] is None and r["burner_lit"] is None


def test_parse_rejects_bool_timer():
    assert parse_reply('{"state":"ok","say":"","timer_seconds":true}')["timer_seconds"] is None


def test_frame_change():
    assert frame_changed(jpeg("white"), jpeg("white")) < 1
    assert frame_changed(jpeg("white"), jpeg("black")) > 100


def test_goal_met_only_for_true():
    assert parse_reply('{"state":"ok","say":"Golden.","timer_seconds":null,"goal_met":true}')["goal_met"] is True
    assert parse_reply('{"state":"ok","say":"","timer_seconds":null,"goal_met":"yes"}')["goal_met"] is False


def test_parse_plan_sorts_limits_and_drops_bad_items():
    p = parse_plan('Sources: x. {"intro":"About 2 hours.","checkpoints":[{"at_minutes":30,"say":"Stir"},{"at_minutes":10,"say":"Check liquid"},'
                   '{"at_minutes":999,"say":"too late"},{"at_minutes":15},{"at_minutes":true,"say":"x"}]}')
    assert [c["at_minutes"] for c in p["checkpoints"]] == [10, 30] and p["intro"] == "About 2 hours."
    assert parse_plan("nope") == {"intro": "", "checkpoints": []}


def test_checkpoints_fire_once_by_the_clock_and_name_the_item():
    s = sess([item("beef stew", [{"at_minutes": 15, "say": "Check the liquid"}, {"at_minutes": 30, "say": "Stir"}])])
    assert due_messages(s, 14 * 60) == []
    assert [m["say"] for m in due_messages(s, 16 * 60)] == ["beef stew: Check the liquid"]
    assert due_messages(s, 17 * 60) == []
    assert [m["say"] for m in due_messages(s, 31 * 60)] == ["beef stew: Stir"]


def test_items_have_independent_clocks():
    s = sess([item("fries", [{"at_minutes": 5, "say": "Check colour"}]), item("cabbage", [{"at_minutes": 5, "say": "Check softness"}], started=600.0)])
    assert [m["say"] for m in due_messages(s, 301)] == ["fries: Check colour"]
    assert [m["say"] for m in due_messages(s, 901)] == ["cabbage: Check softness"]
    assert due_messages(s, 2000) == []


def test_real_mode_stove_timers_are_minutes_not_seconds():
    assert [secs for secs, _ in UNATTENDED_STEPS] == [300, 600, 1200, 1800]


def test_unattended_stove_reminders_escalate_and_reset():
    s = sess(facts=dict(AWAY))
    assert due_messages(s, 100) == []
    assert due_messages(s, 100 + 200) == []                        # 3 minutes away: still quiet
    first = due_messages(s, 100 + 301)
    assert len(first) == 1 and first[0]["state"] == "watch" and first[0]["say"] == MESSAGES["en"]["away_300"]
    assert due_messages(s, 100 + 400) == []
    assert due_messages(s, 100 + 601)[0]["say"] == MESSAGES["en"]["away_600"]
    assert due_messages(s, 100 + 1201)[0]["state"] == "alert"
    s["facts"]["person_present"] = True
    assert due_messages(s, 100 + 1300) == [] and s["away_level"] == 0


def test_demo_mode_compresses_time_but_keeps_the_same_logic():
    s = sess(facts=dict(AWAY), speed=20.0)                         # 300/20 = 15 s, 600/20 = 30 s
    assert due_messages(s, 100) == [] and due_messages(s, 110) == []
    assert [m["say"] for m in due_messages(s, 116)] == [MESSAGES["en"]["away_300"]]
    assert due_messages(s, 131)[0]["say"] == MESSAGES["en"]["away_600"]


def test_demo_mode_also_compresses_cooking_checkpoints():
    s = sess([item("eggs", [{"at_minutes": 10, "say": "Lower the heat"}])], speed=20.0)
    assert due_messages(s, 29) == [] and [m["say"] for m in due_messages(s, 31)] == ["eggs: Lower the heat"]


def test_kiswahili_alerts_use_the_kiswahili_messages():
    s = sess(facts=dict(AWAY), lang="sw")
    due_messages(s, 100)
    assert due_messages(s, 100 + 301)[0]["say"] == MESSAGES["sw"]["away_300"]


def test_every_stove_step_has_a_message_in_both_languages():
    for secs, _ in UNATTENDED_STEPS:
        assert f"away_{secs}" in MESSAGES["en"] and f"away_{secs}" in MESSAGES["sw"]


def test_prompts_format_without_errors_and_keep_the_key_rules():
    p = SYSTEM_PROMPT.format(food="fries", language="English", goal_line="g", last_say_line="l")
    assert "NEVER narrate" in p and '"goal_met"' in p
    q = PLAN_PROMPT.format(food="eggs", pref="soft", language="English")
    assert "do not spoil" in q or "not spoil the food" in q and "never what is inside the food" in q


def test_no_unattended_reminder_when_pan_not_visible_or_burner_unknown():
    assert due_messages(sess(facts={"burner_lit": True, "person_present": False, "pan_visible": False}), 9999) == []
    assert due_messages(sess(facts={"person_present": False}), 9999) == []


def test_output_text_variants():
    assert _output_text(SimpleNamespace(output_text="hello")) == "hello"
    part = SimpleNamespace(text="a"); other = SimpleNamespace(text="b")
    assert _output_text(SimpleNamespace(output_text=None, output=[SimpleNamespace(content=[part, other])])) == "ab"
    assert _output_text(SimpleNamespace()) == ""


def test_make_plan_uses_search_then_falls_back_to_memory():
    good = '{"intro":"About 8 minutes (estimate).","checkpoints":[{"at_minutes":4,"say":"Check the shells"}]}'
    saved = (app_main._grounded_text, app_main._chat)
    try:
        app_main._grounded_text = lambda prompt: good
        app_main._chat = lambda prompt, shrunk=(): "should not be used"
        assert make_plan("boiled eggs", "", "English")["source"] == "web"

        def broken(prompt): raise RuntimeError("search failed")
        app_main._grounded_text = broken
        app_main._chat = lambda prompt, shrunk=(): good
        p = make_plan("boiled eggs", "", "English")
        assert p["source"] == "memory" and p["checkpoints"][0]["at_minutes"] == 4

        app_main._grounded_text = lambda prompt: "no json here"
        assert make_plan("boiled eggs", "", "English")["source"] == "memory"
    finally:
        app_main._grounded_text, app_main._chat = saved

def test_parse_timer():
    assert parse_timer("Set a timer for 5 minutes") == 300 and parse_timer("remind me in ten min") == 600
    assert parse_timer("timer for 30 seconds") == 30 and parse_timer("set a timer for an hour") == 3600
    assert parse_timer("how long will the rice take") is None and parse_timer("") is None


def test_voice_commands_are_routed():
    saved = (app_main.call_model, app_main.new_item)
    try:
        app_main.call_model = lambda prompt, imgs: {"state": "ok", "say": "About 10 minutes (estimate).", "timer_seconds": None, "goal_met": False}
        app_main.new_item = lambda name, pref, lang, now: {"name": name, "started": now, "plan_idx": 0, "intro": "Guide.", "checkpoints": [], "source": "web"}
        s = {"language_code": "en", "language": "English", "items": [], "food": "eggs", "last": b"frame"}
        assert handle_utterance(s, "No thanks.")["kind"] == "decline"
        r = handle_utterance(s, "Set a timer for 5 minutes")
        assert r["kind"] == "timer" and r["timer_seconds"] == 300
        r = handle_utterance(s, "Add cabbage")
        assert r["kind"] == "add_item" and r["item"]["name"] == "cabbage" and s["items"][0]["name"] == "cabbage" and s["food"] == "cabbage"
        r = handle_utterance(s, "How long will the rice take?")
        assert r["kind"] == "answer" and r["say"].startswith("About 10")
        assert handle_utterance(s, "")["kind"] == "empty"
        s["last"] = None
        assert handle_utterance(s, "is it ready")["kind"] == "empty"
        assert handle_utterance({**s, "language_code": "sw"}, "no")["say"] == MESSAGES["sw"]["quiet"]
    finally:
        app_main.call_model, app_main.new_item = saved


def test_usage_cost_is_tracked_per_session():
    u = new_usage(); USAGE.set(u)
    _add_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=100_000)), "muse-spark-1.3")
    _add_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0)), "muse-wonder-1.0")
    _add_usage(SimpleNamespace(), "muse-spark-1.3")                      # no usage block: ignored
    assert u["calls"] == 2 and abs(u["usd"] - (1.25 + 0.425 + 4.0)) < 1e-9
    u["voice_s"] = 3600
    assert abs(usage_view(u)["usd"] - (5.675 + 0.18)) < 1e-6
    USAGE.set(None)
    _add_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=5)), "muse-spark-1.3")   # no session: ignored


def test_wake_word_is_required_and_forgiving():
    assert strip_wake("Mpishi, add cabbage") == (True, "add cabbage")
    assert strip_wake("hey Mishi set a timer for 5 minutes") == (True, "set a timer for 5 minutes")
    assert strip_wake("Pishi") == (True, "")
    assert strip_wake("we need more onions") == (False, "") and strip_wake("hey, add cabbage") == (False, "")
    assert strip_wake("") == (False, "")