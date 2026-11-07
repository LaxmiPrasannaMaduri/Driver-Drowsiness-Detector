"""Real-time Driver Drowsiness Detection website.

Run:  python app.py      then open  http://localhost:5001
"""
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

from DetectorCore import CAMERA_INDEX, DrowsinessDetector

PORT = int(os.getenv("PORT", 5001))
app = Flask(__name__)
lock = threading.Lock()
state = {"jpg": None, "info": {"status": "STARTING", "ear": 0, "mar": 0, "yawns": 0, "closed_for": 0}}


def capture_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera. Try CAMERA_INDEX=1")
        return
    detector = DrowsinessDetector()
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        out, info = detector.process(frame)
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with lock:
                state["jpg"], state["info"] = jpg.tobytes(), info


def mjpeg():
    while True:
        with lock:
            jpg = state["jpg"]
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        return jsonify(state["info"])


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Driver Drowsiness Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{margin:0;font-family:system-ui,Arial,sans-serif;background:#0f172a;color:#e2e8f0;text-align:center}
  h1{margin:16px 0 4px;font-size:1.5rem}
  .wrap{display:inline-block;position:relative;border:6px solid #16a34a;border-radius:10px;overflow:hidden;margin-top:10px;transition:border-color .2s}
  .wrap.bad{border-color:#dc2626;animation:flash .6s infinite alternate}
  .wrap.warn{border-color:#f59e0b}
  @keyframes flash{from{box-shadow:0 0 0 0 #dc2626}to{box-shadow:0 0 40px 10px #dc2626}}
  img{display:block;max-width:100%;width:760px}
  #badge{display:inline-block;margin:14px 0 6px;padding:10px 28px;border-radius:8px;font-size:1.8rem;font-weight:700;background:#16a34a;color:#fff}
  #badge.bad{background:#dc2626}#badge.warn{background:#f59e0b;color:#111}#badge.idle{background:#475569}
  .stats{display:flex;gap:12px;justify-content:center;margin:8px;flex-wrap:wrap}
  .stat{background:#1e293b;padding:8px 16px;border-radius:8px;min-width:90px}
  .stat b{display:block;font-size:1.2rem}
  button{margin:8px;padding:8px 16px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-size:1rem;cursor:pointer}
</style></head>
<body>
  <h1>Driver Drowsiness Detection &ndash; Live</h1>
  <div id="wrap" class="wrap"><img src="/video_feed" alt="live camera"></div><br>
  <div id="badge" class="idle">STARTING...</div>
  <div class="stats">
    <div class="stat">Eye ratio<b id="ear">-</b></div>
    <div class="stat">Mouth ratio<b id="mar">-</b></div>
    <div class="stat">Eyes closed (s)<b id="closed">-</b></div>
    <div class="stat">Yawns / min<b id="yawns">-</b></div>
  </div>
  <button id="snd">Enable alarm sound</button>
<script>
let soundOn = false, ctx = null, lastBeep = 0;
document.getElementById('snd').onclick = e => {
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  soundOn = true; e.target.textContent = 'Alarm sound enabled';
};
function beep(){
  if(!soundOn) return;
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = 1800; o.connect(g); g.connect(ctx.destination);
  g.gain.value = 0.2; o.start(); o.stop(ctx.currentTime + 0.35);
}
async function poll(){
  try{
    const s = await (await fetch('/status')).json();
    const bad = s.status==='DROWSY' || s.status==='SLEEPING';
    const warn = s.status==='YAWNING' || s.status==='CALIBRATING';
    const badge = document.getElementById('badge');
    badge.textContent = bad ? s.status + '!' : s.status;
    badge.className = bad ? 'bad' : warn ? 'warn' : (s.status==='ALERT' ? '' : 'idle');
    document.getElementById('wrap').className = 'wrap ' + (bad ? 'bad' : warn ? 'warn' : '');
    document.getElementById('ear').textContent = Number(s.ear).toFixed(2);
    document.getElementById('mar').textContent = Number(s.mar).toFixed(2);
    document.getElementById('closed').textContent = s.closed_for;
    document.getElementById('yawns').textContent = s.yawns;
    if(bad && Date.now()-lastBeep > 1000){ beep(); lastBeep = Date.now(); }
  }catch(e){}
  setTimeout(poll, 150);
}
poll();
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    print(f"Open http://localhost:{PORT} in your browser")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)