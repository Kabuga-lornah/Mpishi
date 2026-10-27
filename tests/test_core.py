import io
from PIL import Image
from main import parse_reply, frame_changed


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), color).save(buf, "JPEG")
    return buf.getvalue()


def test_parse_good():
    r = parse_reply('```json\n{"state":"alert","say":"Smoke. Turn it off.","timer_seconds":300}\n```')
    assert r == {"state": "alert", "say": "Smoke. Turn it off.", "timer_seconds": 300, "goal_met": False}


def test_parse_garbage_is_safe():
    assert parse_reply("not json")["state"] == "unsure"
    assert parse_reply(None)["state"] == "unsure"
    assert parse_reply('{"state":"boom","say":5,"timer_seconds":-4}') == {"state": "unsure", "say": "", "timer_seconds": None, "goal_met": False}


def test_parse_rejects_bool_timer():
    assert parse_reply('{"state":"ok","say":"","timer_seconds":true}')["timer_seconds"] is None


def test_frame_change():
    assert frame_changed(jpeg("white"), jpeg("white")) < 1
    assert frame_changed(jpeg("white"), jpeg("black")) > 100


def test_goal_met_only_for_true():
    assert parse_reply('{"state":"ok","say":"Golden.","timer_seconds":null,"goal_met":true}')["goal_met"] is True
    assert parse_reply('{"state":"ok","say":"","timer_seconds":null,"goal_met":"yes"}')["goal_met"] is False
