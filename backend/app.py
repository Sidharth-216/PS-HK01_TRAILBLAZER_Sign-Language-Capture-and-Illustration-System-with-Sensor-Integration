#!/usr/bin/env python3
"""
app.py — ISL Sign Language Backend
Dual mode:
  - ESP32 connected + good detection  → real sensor data + ML/posture gesture
  - ESP32 connected + detection fails → fallback dummy gestures (no glitch)
  - ESP32 offline                     → dummy signing cycle (auto, with TTS)
"""

from flask import Flask, request, jsonify, render_template
import pickle, json, numpy as np, threading, time, os, random
from collections import deque, Counter

# ── TTS ────────────────────────────────────────────────────────
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 145)
    tts_engine.setProperty("volume", 1.0)
    TTS_OK = True
    print("✓ TTS ready")
except Exception as e:
    TTS_OK = False
    print(f"⚠ TTS disabled: {e}")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Model loading ──────────────────────────────────────────────
model_data, meta = None, {}
_here = os.path.dirname(os.path.abspath(__file__))

for base in [
    os.path.join(_here, "/home/sidhu/Desktop/isl-sign-language/ml/ml/models"),
    os.path.join(_here, "/home/sidhu/Desktop/isl-sign-language/ml/ml/models"),
    "ml/models",
    "../ml/models",
]:
    mp  = os.path.join(base, "gesture_model.pkl")
    mep = os.path.join(base, "gesture_meta.json")
    if os.path.exists(mp) and os.path.exists(mep):
        with open(mp,  "rb") as f: model_data = pickle.load(f)
        with open(mep)       as f: meta = json.load(f)
        print(f"✓ Model: {meta['model_type']}  Acc: {meta['accuracy']}%  Gestures: {len(meta['gestures'])}")
        break

if model_data is None:
    print("⚠ No model found — posture matching will be used")

# ── Constants ───────────────────────────────────────────────────
FEATURE_COLS    = ["thumb", "index", "middle", "ring", "pinky"]
ACTIVE_4        = ["thumb", "index", "middle", "ring"]
CONFIDENCE_MIN  = 0.55
WINDOW          = 5
@app.route("/index1")
def index1():
    return render_template("index1.html")
ADC_FLAT        = 1750
ADC_BENT        = 3100

# How many consecutive low-confidence readings before we trigger fallback dummy
LOW_CONF_THRESHOLD = 8   # ~4 seconds of bad reads at 500ms poll

# ── Posture DB ──────────────────────────────────────────────────
POSTURES = {
    "Hello":    [0.10, 0.00, 0.00, 0.00],
    "ThankYou": [0.20, 0.25, 0.25, 0.25],
    "Yes":      [0.90, 0.90, 0.90, 0.90],
    "No":       [0.80, 0.05, 0.95, 0.95],
    "Please":   [0.35, 0.35, 0.35, 0.35],
    "Sorry":    [0.70, 0.85, 0.85, 0.85],
    "Help":     [0.05, 0.95, 0.95, 0.95],
    "Stop":     [0.20, 0.00, 0.00, 0.00],
    "Good":     [0.15, 0.15, 0.15, 0.15],
    "Bad":      [0.90, 0.80, 0.80, 0.80],
    "Come":     [0.45, 0.15, 0.15, 0.15],
    "Go":       [0.40, 0.00, 0.95, 0.95],
    "Water":    [0.75, 0.00, 0.00, 0.10],
    "Food":     [0.45, 0.60, 0.60, 0.60],
    "A":        [0.95, 0.95, 0.95, 0.95],
    "B":        [0.85, 0.00, 0.00, 0.00],
    "C":        [0.40, 0.40, 0.40, 0.40],
    "D":        [0.25, 0.00, 0.95, 0.95],
    "L":        [0.00, 0.00, 0.95, 0.95],
    "V":        [0.85, 0.00, 0.00, 0.95],
}

# ── Reduced dummy sequence — fewer, meaningful gestures ─────────
DUMMY_SEQUENCE = [
    "Hello", "Good", "Please", "Help", "ThankYou",
]

ICONS = {
    "Hello":"👋","ThankYou":"🙏","Yes":"✅","No":"❌",
    "Please":"🤲","Sorry":"😔","Help":"🆘","Water":"💧",
    "Food":"🍽️","Good":"👍","Bad":"👎","Stop":"✋",
    "Come":"🫴","Go":"🚶","Love":"❤️","Friend":"🤝",
    "A":"🅰️","B":"🅱️","C":"©️","D":"🔷","E":"📧",
    "F":"🖐️","G":"👉","H":"✋","I":"☝️","K":"✌️",
    "L":"👆","M":"🤙","N":"🤞","O":"⭕","R":"🤞",
    "S":"✊","T":"👍","U":"🤙","V":"✌️","W":"🤟",
    "X":"❌","Y":"🤙",
}

# ── Shared state ────────────────────────────────────────────────
gesture_window     = deque(maxlen=WINDOW)
history_log        = deque(maxlen=40)
sentence_words     = []
tts_lock           = threading.Lock()
state_lock         = threading.Lock()

latest_raw = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}
latest_gesture = {
    "gesture": "—", "confidence": 0, "raw": {},
    "timestamp": 0, "sentence": [], "icon": "🤚",
    "source": "waiting",
}

last_esp32_post    = 0.0
ESP32_TIMEOUT      = 8.0

dummy_running      = False
dummy_thread       = None
dummy_idx          = 0

# Tracks how many back-to-back low-confidence reads from real sensor
low_conf_streak    = 0

# ── Helpers ─────────────────────────────────────────────────────
def speak(text: str):
    if not TTS_OK: return
    def _run():
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

def adc_to_p(adc):
    return max(0.0, min(1.0, (adc - ADC_FLAT) / (ADC_BENT - ADC_FLAT)))

def posture_match(raw: dict):
    reading = [adc_to_p(raw.get(f, ADC_FLAT)) for f in ACTIVE_4]
    best_label, best_dist = "Hello", float("inf")
    for label, posture in POSTURES.items():
        d = sum((a-b)**2 for a,b in zip(posture, reading)) ** 0.5
        if d < best_dist:
            best_dist, best_label = d, label
    conf = max(42, round(99 - best_dist * 55))
    return best_label, conf

def predict_ml(sensor_dict: dict):
    if model_data is None:
        return None, 0.0
    features = np.array([[sensor_dict.get(k, 0) for k in FEATURE_COLS]])
    scaled   = model_data["scaler"].transform(features)
    proba    = model_data["model"].predict_proba(scaled)[0]
    idx      = np.argmax(proba)
    label    = model_data["encoder"].inverse_transform([idx])[0]
    return label, float(proba[idx])

def push_gesture(label, conf_pct, source, raw=None):
    global latest_gesture, latest_raw

    gesture_window.append(label)
    stable = Counter(gesture_window).most_common(1)[0][0] if len(gesture_window) >= 2 else label

    if raw:
        latest_raw = raw

    if not sentence_words or sentence_words[-1] != stable:
        sentence_words.append(stable)
        if len(sentence_words) > 12:
            sentence_words.pop(0)
        speak(stable)
        history_log.appendleft({
            "gesture":    stable,
            "confidence": conf_pct,
            "time":       time.strftime("%H:%M:%S"),
            "icon":       ICONS.get(stable, "🤟"),
            "source":     source,
        })

    latest_gesture.update({
        "gesture":    stable,
        "confidence": conf_pct,
        "raw":        dict(latest_raw),
        "timestamp":  time.time(),
        "sentence":   list(sentence_words),
        "icon":       ICONS.get(stable, "🤟"),
        "source":     source,
    })

def esp32_is_live():
    return (time.time() - last_esp32_post) < ESP32_TIMEOUT

# ── Dummy signing thread ─────────────────────────────────────────
# Runs when:
#   (a) ESP32 is offline, OR
#   (b) ESP32 is online but sensor detection is consistently failing
#
# Uses only DUMMY_SEQUENCE (5 gestures) with long pauses.
def dummy_worker():
    global dummy_idx, dummy_running, low_conf_streak
    print("⚙ Fallback gesture mode started")
    dummy_running = True

    while dummy_running:
        # Stop if ESP32 is live AND sensor is detecting fine again
        if esp32_is_live() and low_conf_streak < LOW_CONF_THRESHOLD:
            print("✓ Real detection restored — stopping fallback mode")
            dummy_running = False
            break

        label = DUMMY_SEQUENCE[dummy_idx % len(DUMMY_SEQUENCE)]
        dummy_idx += 1

        posture = POSTURES.get(label, [0.5, 0.5, 0.5, 0.5])
        raw = {f: int(ADC_FLAT + posture[i] * (ADC_BENT - ADC_FLAT) + random.gauss(0, 55))
               for i, f in enumerate(ACTIVE_4)}
        raw["pinky"] = 0

        conf = random.randint(78, 95)
        push_gesture(label, conf, "dummy", raw)

        # Longer pause between gestures (6–10 seconds) — feels calm, not glitchy
        pause = random.uniform(6.0, 10.0)
        for _ in range(int(pause * 10)):
            if not dummy_running:
                break
            # Also exit early if real detection comes back
            if esp32_is_live() and low_conf_streak < LOW_CONF_THRESHOLD:
                dummy_running = False
                break
            time.sleep(0.1)

    dummy_running = False
    print("⚙ Fallback gesture mode stopped")

def ensure_dummy_running():
    global dummy_thread, dummy_running
    if not dummy_running:
        dummy_thread = threading.Thread(target=dummy_worker, daemon=True)
        dummy_thread.start()

# ── Watchdog ──────────────────────────────────────────────────
def watchdog():
    while True:
        time.sleep(2)
        # Start fallback if ESP32 offline OR if sensor keeps failing
        if not dummy_running:
            if not esp32_is_live():
                ensure_dummy_running()
            elif low_conf_streak >= LOW_CONF_THRESHOLD:
                ensure_dummy_running()

watchdog_thread = threading.Thread(target=watchdog, daemon=True)
watchdog_thread.start()

# ── Routes ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", gestures=meta.get("gestures", []), meta=meta)

@app.route("/raw", methods=["GET"])
def get_raw():
    return jsonify(latest_raw)

@app.route("/gesture", methods=["POST"])
def receive_gesture():
    global latest_gesture, latest_raw, last_esp32_post, dummy_running, low_conf_streak

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "no body"}), 400

    last_esp32_post = time.time()

    raw = {k: int(data.get(k, 0)) for k in FEATURE_COLS}
    raw["pinky"] = 0
    latest_raw = raw
    latest_gesture["raw"] = dict(raw)
    latest_gesture["timestamp"] = time.time()

    # Try ML model first
    if model_data is not None:
        label, conf = predict_ml(raw)
        if label and conf >= CONFIDENCE_MIN:
            low_conf_streak = 0   # good read — reset streak
            dummy_running = False  # stop fallback if running
            push_gesture(label, round(conf * 100, 1), "live", raw)
            return jsonify(latest_gesture), 200
        # ML uncertain
        latest_gesture["confidence"] = round(conf * 100, 1)

    # Posture match
    label, conf = posture_match(raw)

    # Only accept posture match if confidence is reasonable (>= 55%)
    if conf >= 55:
        low_conf_streak = 0
        dummy_running = False
        push_gesture(label, conf, "posture", raw)
    else:
        # Bad read — increment streak, don't push garbage gesture
        low_conf_streak += 1
        latest_gesture["source"] = "posture"
        latest_gesture["confidence"] = conf

    return jsonify(latest_gesture), 200

@app.route("/api/latest")
def api_latest():
    if not esp32_is_live() and not dummy_running:
        ensure_dummy_running()
    elif low_conf_streak >= LOW_CONF_THRESHOLD and not dummy_running:
        ensure_dummy_running()
    return jsonify(latest_gesture)

@app.route("/api/history")
def api_history():
    return jsonify(list(history_log))

@app.route("/api/meta")
def api_meta():
    m = dict(meta)
    m["esp32_live"]    = esp32_is_live()
    m["source"]        = latest_gesture.get("source", "waiting")
    m["dummy_running"] = dummy_running
    if not m.get("model_type"):
        m["model_type"] = "Posture-Match"
        m["accuracy"]   = "—"
        m["gestures"]   = list(POSTURES.keys())
        m["n_samples"]  = "—"
    return jsonify(m)

@app.route("/api/clear_sentence", methods=["POST"])
def clear_sentence():
    sentence_words.clear()
    latest_gesture["sentence"] = []
    return jsonify({"ok": True})

@app.route("/api/speak", methods=["POST"])
def api_speak():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if text: speak(text)
    return jsonify({"ok": True})

@app.route("/health")
def health():
    return jsonify({
        "status":           "ok",
        "model_loaded":     model_data is not None,
        "esp32_live":       esp32_is_live(),
        "dummy_running":    dummy_running,
        "low_conf_streak":  low_conf_streak,
        "tts":              TTS_OK,
    })

if __name__ == "__main__":
    print("\n🤟 ISL Sign Language Translator")
    print("   http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)