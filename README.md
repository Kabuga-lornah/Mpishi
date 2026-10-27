# Mpishi

An extra set of eyes on your stove. A phone camera watches the pan, Meta's Muse Spark describes what changed,
and Mpishi speaks up on its own (English or Kiswahili). It sets timers and answers "is it ready yet?".
Not a smoke alarm. Never leave hot oil unattended.

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add your Meta Model API key
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 on a laptop with a webcam. Browsers only allow the camera on localhost or https,
so for a phone use an https tunnel to port 8000 (for example `cloudflared tunnel --url http://localhost:8000`).

## Test
`pytest`

## Deploy (Render)
Build: `pip install -r requirements.txt`  Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
Environment: `MODEL_API_KEY` (required), `MPISHI_ACCESS_CODE` (set this so strangers cannot spend your credits).
Free instances sleep after 15 idle minutes; open the link a minute before cooking.
