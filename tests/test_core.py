import io
from types import SimpleNamespace
from PIL import Image
import main as app_main
from main import parse_reply, parse_plan, frame_changed, due_messages, _output_text, make_plan


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), color).save(buf, "JPEG")
    return buf.getvalue()


def item(name, points, started=0.0):
    return {"name": name, "started": started, "plan_idx": 0, "intro": "", "checkpoints": list(points), "source": "web"}


def sess(items=(), facts=None):
    return {"items": list(items), "facts": facts or {}, "away_since": None, "away_level": 0, "language": "English"}


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


def test_unattended_stove_reminders_escalate_and_reset():
    s = sess(facts={"burner_lit": True, "person_present": False, "pan_visible": True})
    assert due_messages(s, 100) == []
    first = due_messages(s, 100 + 301)
    assert len(first) == 1 and "5 minutes" in first[0]["say"] and first[0]["state"] == "watch"
    assert due_messages(s, 100 + 400) == []
    assert "10 minutes" in due_messages(s, 100 + 601)[0]["say"]
    assert due_messages(s, 100 + 1201)[0]["state"] == "alert"
    s["facts"]["person_present"] = True
    assert due_messages(s, 100 + 1300) == [] and s["away_level"] == 0


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