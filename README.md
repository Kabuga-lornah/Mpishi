# Mpishi

An extra set of eyes on your stove. A camera watches the cooking area, Meta's Muse Spark reads what it sees as
structured facts (is a pan visible, is a person near, is a burner lit), and Mpishi speaks up only when the cook
needs to know something. English and Kiswahili. Not a smoke alarm and not a replacement for watching hot oil.

## What it does
- **Silent by default.** It does not narrate the cook. It speaks for safety conditions, cooking checkpoints and questions.
- **Stove left on.** If a burner looks lit and nobody is near, it reminds the cook at 5, 10, 20 and 30 minutes.
- **Cooking guide per item.** Muse Wonder with web search plans each dish (estimates only); items have separate clocks.
- **Demo mode.** A checkbox compresses the timers (default 20x: 15, 30, 60 and 90 seconds) and shows a "Demo mode" badge.
- **Honest failure.** If the camera or connection stops, a red "MPISHI IS NOT WATCHING" banner appears.

## Run (Windows PowerShell)
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then put your Meta Model API key in .env
.venv\Scripts\python -m uvicorn main:app --port 8000
```
Open http://localhost:8000. Browsers only allow the camera on localhost or https. For a phone, use an https tunnel to
port 8000 (for example `cloudflared tunnel --url http://localhost:8000`) and set `MPISHI_ACCESS_CODE` first.

## Access code
Leave `MPISHI_ACCESS_CODE` empty for local use. If it is set, every API call needs it and the page shows an access-code box.
Set it before sharing a tunnel or deploying, so strangers cannot spend your credits.

## Test
`.venv\Scripts\python -m pytest -q`

## Known limits
- A camera cannot see inside food, read oil temperature, or reliably see every flame or smoke. Mpishi says so and asks the cook to check.
- Cooking times are web-sourced estimates for a home stove. Always check by hand or with a thermometer.
- Sessions live in memory and are lost when the server restarts. Free hosting sleeps when idle, so demo locally.
- The Kiswahili text has not been reviewed by a native speaker.