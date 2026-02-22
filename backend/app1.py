#!/usr/bin/env python3
"""
ISL Sign Language Unified Backend
==================================
Combines TWO input modes:
  MODE A — Webcam (MediaPipe hand landmarks + rule-based ML)
  MODE B — ESP32 Glove (flex sensors → posture/ML classifier)

Features:
  - Real-time gesture detection from both sources
  - Gemini AI for translation + chat
  - TTS voice output for detected gestures (Web Speech API on frontend)
  - Unified sentence builder
  - History and session tracking

ESP32 main.py compatibility fixes applied:
  - ESP32 sends all 5 fingers INCLUDING "pinky" — pinky is now READ (not zeroed)
  - ESP32 sends extra "ts" field — ignored safely (not parsed as a sensor)
  - Posture table expanded to 5-finger vectors to match real sensor data
  - ADC_FLAT/ADC_BENT tuned to 12-bit full-range (ATTN_11DB): flat≈1500, bent≈3500
  - posture_match() now uses all 5 fingers for better sign discrimination
  - low_conf_streak threshold raised to 15 to avoid triggering fallback prematurely
  - dummy_running cleared BEFORE state_lock in success branch to beat the watchdog
  - /api/calibrate endpoint added to read live ADC values for tuning calibration
"""

import os, json, time, base64, random, threading, sys
from collections import deque, Counter
from flask import Flask, request, jsonify, render_template

import numpy as np
import cv2
import google.generativeai as genai

# Add parent directory to path to import ml_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import mediapipe as mp
    mp_hands           = mp.solutions.hands
    mp_drawing         = mp.solutions.drawing_utils
    mp_drawing_styles  = mp.solutions.drawing_styles
    hands_detector     = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence  = 0.5,
    )
    MEDIAPIPE_OK = True
    print("✓ MediaPipe loaded")
except Exception as e:
    MEDIAPIPE_OK = False
    print(f"⚠ MediaPipe not available: {e}")

try:
    from ml_model.gesture_model import GestureRecognizer, ISL_SIGNS
except ImportError:
    print("⚠ Could not import gesture_model from ml_model. Running with limited gesture support.")
    # Fallback: define a minimal GestureRecognizer
    class GestureRecognizer:
        def predict(self, landmarks):
            return None, 0.0
        def get_sign_reference(self):
            return {}
    ISL_SIGNS = {}

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Gemini ──────────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "xxx")
GEMINI_AVAILABLE = True
gemini_model     = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model     = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
        print("✓ Gemini AI loaded")
    except Exception as e:
        print(f"⚠ Gemini failed: {e}")

# ── Gesture Model ───────────────────────────────────────────────
gesture_recognizer = GestureRecognizer()

# ── ESP32 / Glove Constants ─────────────────────────────────────
#
# HOW TO CALIBRATE YOUR FLEX SENSORS:
#   1. Run server, connect glove.
#   2. Hold all fingers STRAIGHT → GET /api/calibrate → note avg values → set ADC_FLAT
#   3. Curl all fingers FULLY   → GET /api/calibrate → note avg values → set ADC_BENT
#   4. Restart the server.
#
# ESP32 sends (from main.py post_gesture):
#   {"thumb": int, "index": int, "middle": int, "ring": int, "pinky": int, "ts": int}
#   "ts" is ignored below.
#
FEATURE_COLS    = ["thumb", "index", "middle", "ring", "pinky"]
ADC_FLAT        = 1500   # ADC reading when finger is STRAIGHT — tune to your sensors
ADC_BENT        = 3500   # ADC reading when finger is CURLED  — tune to your sensors
ADC_RANGE       = ADC_BENT - ADC_FLAT
CONFIDENCE_MIN  = 0.55
WINDOW          = 5
ESP32_TIMEOUT   = 8.0
LOW_CONF_THRESH = 15     # require 15 consecutive bad reads before triggering fallback

# Posture table: [thumb, index, middle, ring, pinky] — 5 fingers, 0=straight 1=curled
# FIX: old code only had 4 values and zeroed pinky, breaking signs that use the pinky
POSTURES = {
    "Hello"   : [0.10, 0.00, 0.00, 0.00, 0.00],
    "ThankYou": [0.20, 0.25, 0.25, 0.25, 0.20],
    "Yes"     : [0.90, 0.90, 0.90, 0.90, 0.90],
    "No"      : [0.80, 0.05, 0.95, 0.95, 0.95],
    "Please"  : [0.35, 0.35, 0.35, 0.35, 0.35],
    "Sorry"   : [0.70, 0.85, 0.85, 0.85, 0.85],
    "Help"    : [0.05, 0.95, 0.95, 0.95, 0.95],
    "Stop"    : [0.20, 0.00, 0.00, 0.00, 0.00],
    "Good"    : [0.15, 0.15, 0.15, 0.15, 0.15],
    "Bad"     : [0.90, 0.80, 0.80, 0.80, 0.80],
    "Water"   : [0.75, 0.00, 0.00, 0.10, 0.00],
    "Food"    : [0.45, 0.60, 0.60, 0.60, 0.60],
    "A"       : [0.95, 0.95, 0.95, 0.95, 0.95],  # fist
    "B"       : [0.85, 0.00, 0.00, 0.00, 0.00],  # thumb tucked, all fingers flat
    "C"       : [0.40, 0.40, 0.40, 0.40, 0.40],  # all half-curled
    "D"       : [0.30, 0.00, 0.90, 0.90, 0.90],  # index up, others curled
    "V"       : [0.90, 0.00, 0.00, 0.90, 0.90],  # index + middle up
    "L"       : [0.00, 0.00, 0.90, 0.90, 0.90],  # index + thumb out
    "Y"       : [0.00, 0.90, 0.90, 0.90, 0.00],  # thumb + pinky out
    "I"       : [0.90, 0.90, 0.90, 0.90, 0.00],  # only pinky extended
}

ICONS = {
    "Hello":"👋","ThankYou":"🙏","Yes":"✅","No":"❌","Please":"🤲",
    "Sorry":"😔","Help":"🆘","Water":"💧","Food":"🍽️","Good":"👍",
    "Bad":"👎","Stop":"✋","Love":"❤️","A":"🅰️","B":"🅱️","C":"©️",
    "D":"☝️","V":"✌️","L":"🤙","W":"🖖","Y":"🤙","I":"🤙",
    "ONE":"1️⃣","TWO":"2️⃣","THREE":"3️⃣","FOUR":"4️⃣","FIVE":"5️⃣",
    "HELLO":"👋","THANK YOU":"🙏","PLEASE":"🤲","YES":"✅","NO":"❌",
    "HELP":"🆘","WATER":"💧","FOOD":"🍽️","GOOD":"👍","BAD":"👎",
    "NAMASTE":"🙏",
}

DUMMY_SEQUENCE = ["Hello", "Good", "Please", "Help", "ThankYou", "Water", "Food"]

# ── Shared State ─────────────────────────────────────────────────
state_lock = threading.Lock()

glove_raw   = {k: 0 for k in FEATURE_COLS}
glove_state = {
    "gesture"   : "—",
    "confidence": 0,
    "raw"       : {},
    "timestamp" : 0,
    "sentence"  : [],
    "icon"      : "🤚",
    "source"    : "waiting",
}

last_esp32_post  = 0.0
low_conf_streak  = 0
dummy_running    = False
dummy_thread     = None
dummy_idx        = 0

# Calibration buffer — stores last 20 raw ADC dicts for /api/calibrate
_calibration_buf = deque(maxlen=20)

unified = {
    "last_gesture"    : None,
    "last_gesture_raw": None,
    "last_add_time"   : 0,
    "sentence"        : [],
    "history"         : [],
    "tts_queue"       : [],
}


# ── ESP32 / Glove Helpers ────────────────────────────────────────

def adc_to_flex(adc_val: int) -> float:
    """Convert raw 12-bit ADC int → normalised flex fraction [0.0 … 1.0].
    0.0 = finger straight, 1.0 = finger fully curled."""
    return max(0.0, min(1.0, (adc_val - ADC_FLAT) / ADC_RANGE))


def posture_match(raw: dict):
    """
    5-finger Euclidean nearest-neighbour match against POSTURES.

    FIX: old code called adc_to_p() with ADC_FLAT=1750 / ADC_BENT=3100 on
    only 4 fingers (ACTIVE_4), then hardcoded pinky=0. This meant:
      a) real pinky data was discarded
      b) signs distinguished mainly by pinky (I, Y) would mis-classify
      c) the ADC range didn't match ESP32's actual 12-bit ATTN_11DB output
    """
    reading = [adc_to_flex(raw.get(f, ADC_FLAT)) for f in FEATURE_COLS]

    best_label = "Hello"
    best_dist  = float("inf")
    for label, posture in POSTURES.items():
        # Pad to 5 values for backward compat if any posture is still 4-long
        p = posture + [0.0] * (5 - len(posture))
        d = sum((a - b) ** 2 for a, b in zip(reading, p)) ** 0.5
        if d < best_dist:
            best_dist, best_label = d, label

    # Scale: dist=0→99, dist≈√5≈2.24 (theoretical max)→~0; clamp to 35 floor
    conf = max(35, round(99 - best_dist * 45))
    return best_label, conf


def esp32_is_live() -> bool:
    return (time.time() - last_esp32_post) < ESP32_TIMEOUT


def push_to_unified(label: str, confidence, source: str, icon: str = None):
    """Thread-safe: add confirmed gesture to sentence + history + TTS queue."""
    global unified
    if not label or label in ("—", "UNKNOWN", "NONE"):
        return

    with state_lock:
        now    = time.time()
        last_t = unified["last_add_time"]
        last_g = unified["last_gesture_raw"]

        if label == last_g and (now - last_t) < 2.0:
            return   # debounce

        unified["last_gesture_raw"] = label
        unified["last_add_time"]    = now
        unified["last_gesture"]     = label

        if not unified["sentence"] or unified["sentence"][-1] != label:
            unified["sentence"].append(label)
            if len(unified["sentence"]) > 20:
                unified["sentence"] = unified["sentence"][-20:]

            conf_pct = round(confidence * 100 if confidence <= 1.0 else confidence, 1)
            unified["history"].insert(0, {
                "gesture"   : label,
                "confidence": conf_pct,
                "timestamp" : time.strftime("%H:%M:%S"),
                "icon"      : icon or ICONS.get(label, "🤟"),
                "source"    : source,
            })
            if len(unified["history"]) > 50:
                unified["history"] = unified["history"][:50]

            unified["tts_queue"].append(label)
            if len(unified["tts_queue"]) > 10:
                unified["tts_queue"] = unified["tts_queue"][-10:]


# ── MediaPipe / Webcam Helpers ───────────────────────────────────

def extract_hand_landmarks(frame):
    if not MEDIAPIPE_OK:
        return None, []
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            lm = []
            for point in hand_lm.landmark:
                lm.extend([point.x, point.y, point.z])
            landmarks_list.append(lm)
    return results, landmarks_list


def draw_styled_landmarks(frame, results):
    if not MEDIAPIPE_OK or not results or not results.multi_hand_landmarks:
        return frame
    for hand_lm in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
    return frame


def add_hud_overlay(frame, gesture, confidence, sentence):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),      (w, 60),  (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 70), (w, h),   (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "ISL RECOGNITION", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
    if confidence:
        cc = (0,255,100) if confidence > 0.8 else (0,200,255) if confidence > 0.6 else (0,100,255)
        cv2.putText(frame, f"CONF: {confidence:.1%}", (w - 160, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, 1)
    if gesture:
        cv2.putText(frame, str(gesture), (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
    sent_text = " ".join(sentence[-6:]) if sentence else "Start signing..."
    cv2.putText(frame, sent_text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    color, length = (0, 255, 180), 25
    for (x, y) in [(0, 0), (w, 0), (0, h), (w, h)]:
        sx = 1 if x == 0 else -1
        sy = 1 if y == 0 else -1
        cv2.line(frame, (x, y), (x + sx * length, y), color, 2)
        cv2.line(frame, (x, y), (x, y + sy * length), color, 2)
    return frame


# ── Dummy / Fallback Thread ──────────────────────────────────────

def dummy_worker():
    global dummy_running, dummy_idx, low_conf_streak
    print("⚙ Fallback gesture mode started")
    dummy_running = True
    while dummy_running:
        if esp32_is_live() and low_conf_streak < LOW_CONF_THRESH:
            print("✓ Real detection OK — stopping fallback")
            dummy_running = False
            break
        label = DUMMY_SEQUENCE[dummy_idx % len(DUMMY_SEQUENCE)]
        dummy_idx += 1
        posture = POSTURES.get(label, [0.5] * 5)
        p5      = posture + [0.0] * (5 - len(posture))
        raw     = {f: int(ADC_FLAT + p5[i] * ADC_RANGE + random.gauss(0, 55))
                   for i, f in enumerate(FEATURE_COLS)}
        conf    = random.randint(75, 92)
        with state_lock:
            glove_state.update({
                "gesture"   : label,
                "confidence": conf,
                "raw"       : dict(raw),
                "timestamp" : time.time(),
                "icon"      : ICONS.get(label, "🤟"),
                "source"    : "fallback",
            })
        push_to_unified(label, conf, "glove-fallback", ICONS.get(label, "🤟"))
        pause = random.uniform(6.0, 10.0)
        for _ in range(int(pause * 10)):
            if not dummy_running:
                break
            if esp32_is_live() and low_conf_streak < LOW_CONF_THRESH:
                dummy_running = False
                break
            time.sleep(0.1)
    dummy_running = False
    print("⚙ Fallback mode stopped")


def ensure_dummy():
    global dummy_thread, dummy_running
    if not dummy_running:
        dummy_thread = threading.Thread(target=dummy_worker, daemon=True)
        dummy_thread.start()


def watchdog():
    while True:
        time.sleep(2)
        if not dummy_running:
            if not esp32_is_live() or low_conf_streak >= LOW_CONF_THRESH:
                ensure_dummy()

threading.Thread(target=watchdog, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════



# ── MODE A: Webcam ─────────────────────────────────────────────

@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data     = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        img_data = data.get("image", "")
        if not img_data or not isinstance(img_data, str):
            return jsonify({"error": "No image data"}), 400

        # Extract base64 safely
        try:
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
        except Exception as e:
            return jsonify({"error": f"Base64 decode error: {e}"}), 400

        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return jsonify({"error": "Frame decode failed"}), 400

        results, landmarks_list = extract_hand_landmarks(frame)
        if results:
            frame = draw_styled_landmarks(frame, results)

        gesture, confidence = None, 0.0
        hand_detected       = len(landmarks_list) > 0

        if landmarks_list:
            gesture, confidence = gesture_recognizer.predict(landmarks_list[0])
            if gesture and confidence > 0.6 and gesture not in ("UNKNOWN", "NONE"):
                push_to_unified(gesture, confidence, "webcam", ICONS.get(gesture, "🤟"))

        with state_lock:
            frame = add_hud_overlay(frame, gesture, confidence, unified["sentence"])

        _, buffer     = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        landmark_data = []
        if results and results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                landmark_data.append([{"x": lm.x, "y": lm.y} for lm in hand_lm.landmark])

        with state_lock:
            tts_q               = unified["tts_queue"][:]
            unified["tts_queue"] = []
            return jsonify({
                "processed_image": processed_b64,
                "gesture"        : gesture,
                "confidence"     : round(confidence, 3),
                "hand_detected"  : hand_detected,
                "landmarks"      : landmark_data,
                "tts_queue"      : tts_q,
            })
    except Exception as e:
        print(f"❌ process_frame error: {e}")
        return jsonify({"error": str(e)}), 500


# ── MODE B: ESP32 Glove ────────────────────────────────────────

@app.route("/gesture", methods=["POST"])
def receive_glove_gesture():
    """
    Receives the JSON POSTed by ESP32 main.py → post_gesture().

    Payload from ESP32:
      {"thumb": <int>, "index": <int>, "middle": <int>,
       "ring":  <int>, "pinky": <int>, "ts": <int>}

    Fixes vs old code:
      1. "ts" key is IGNORED — it is not a sensor, don't try to cast it.
      2. "pinky" is NOW READ from real sensor (old code replaced it with 0).
      3. Posture vectors updated to 5-finger to match real data.
      4. dummy_running cleared BEFORE acquiring state_lock on success path
         so the watchdog thread cannot restart dummy between the confidence
         check and the flag write.
      5. low_conf_streak only increments on genuinely bad reads; a borderline
         read that passes the 55% threshold resets the streak to 0.
      6. Missing or non-integer sensor values default to ADC_FLAT (straight)
         rather than 0 (which would look like an extreme bend to posture_match).
    """
    global last_esp32_post, dummy_running, low_conf_streak, glove_raw, glove_state

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "no body"}), 400

    # 1. Mark ESP32 as alive
    last_esp32_post = time.time()

    # 2. Extract exactly the 5 finger keys; ignore "ts" and any other fields
    raw = {}
    for finger in FEATURE_COLS:
        val = data.get(finger)
        if val is None:
            raw[finger] = ADC_FLAT   # default to "straight" not 0
            print(f"⚠ Missing '{finger}' in ESP32 payload — using ADC_FLAT")
        else:
            try:
                raw[finger] = int(val)
            except (TypeError, ValueError):
                raw[finger] = ADC_FLAT
                print(f"⚠ Bad value for '{finger}': {val!r} — using ADC_FLAT")

    # 3. Buffer for calibration endpoint
    _calibration_buf.append(dict(raw))

    # 4. Update shared raw view (dashboard sensor bars + waveform)
    with state_lock:
        glove_raw = dict(raw)

    # 5. Run 5-finger posture match
    label, conf = posture_match(raw)
    print(f"[GLOVE] {raw}  →  {label} ({conf}%)")

    if conf >= 55:
        # ── Good reading ──────────────────────────────────────────
        # Clear dummy flag FIRST (before lock) so watchdog can't restart it
        dummy_running   = False
        low_conf_streak = 0

        source = "glove-posture"
        icon   = ICONS.get(label, "🤟")

        with state_lock:
            glove_state.update({
                "gesture"   : label,
                "confidence": conf,
                "raw"       : dict(raw),
                "timestamp" : time.time(),
                "icon"      : icon,
                "source"    : source,
            })

        push_to_unified(label, conf, source, icon)

    else:
        # ── Low-confidence read — don't push bad data to sentence ─
        low_conf_streak += 1
        print(f"⚠ Low confidence ({conf}%) — streak={low_conf_streak}")
        with state_lock:
            glove_state["confidence"] = conf
            glove_state["source"]     = "glove-posture"

    with state_lock:
        return jsonify(glove_state), 200


# ── Calibration ────────────────────────────────────────────────

@app.route("/api/calibrate", methods=["GET"])
def api_calibrate():
    """
    Returns average ADC values from the last 20 glove readings.
    Use to find your real ADC_FLAT (fingers straight) and ADC_BENT (fingers curled).
    """
    buf = list(_calibration_buf)
    if not buf:
        return jsonify({
            "message": (
                "No readings yet. Make sure the ESP32 is connected "
                "and posting to /gesture (PIR must detect motion)."
            )
        })
    avgs = {}
    for finger in FEATURE_COLS:
        vals = [r[finger] for r in buf if finger in r]
        avgs[finger] = round(sum(vals) / len(vals)) if vals else 0

    return jsonify({
        "message"         : f"Averages over last {len(buf)} readings",
        "averages"        : avgs,
        "current_ADC_FLAT": ADC_FLAT,
        "current_ADC_BENT": ADC_BENT,
        "tip"             : "Hold fingers STRAIGHT and call /api/calibrate → set ADC_FLAT. Curl fingers FULLY → set ADC_BENT.",
        "raw_readings"    : buf,
    })


# ── Unified API ────────────────────────────────────────────────

@app.route("/api/unified")
def api_unified():
    with state_lock:
        tts_q               = unified["tts_queue"][:]
        unified["tts_queue"] = []
        return jsonify({
            "sentence"    : unified["sentence"],
            "last_gesture": unified["last_gesture"],
            "history"     : unified["history"][:15],
            "glove"       : {
                "gesture"      : glove_state["gesture"],
                "confidence"   : glove_state["confidence"],
                "icon"         : glove_state["icon"],
                "source"       : glove_state["source"],
                "raw"          : glove_raw,
                "esp32_live"   : esp32_is_live(),
                "dummy_running": dummy_running,
            },
            "tts_queue"   : tts_q,
        })


@app.route("/api/status")
def api_status():
    return jsonify({
        "mediapipe"       : MEDIAPIPE_OK,
        "gemini"          : GEMINI_AVAILABLE,
        "esp32_live"      : esp32_is_live(),
        "dummy_running"   : dummy_running,
        "low_conf_streak" : low_conf_streak,
        "signs_count"     : len(ISL_SIGNS),
        "adc_calibration" : {"flat": ADC_FLAT, "bent": ADC_BENT},
    })


@app.route("/api/clear_sentence", methods=["POST"])
def clear_sentence():
    with state_lock:
        unified["sentence"]     = []
        unified["last_gesture"] = None
        glove_state["sentence"] = []
    return jsonify({"ok": True})


@app.route("/api/history")
def api_history():
    with state_lock:
        return jsonify(unified["history"])


@app.route("/get_isl_signs")
def get_isl_signs():
    return jsonify(gesture_recognizer.get_sign_reference())


# ── Gemini Routes ──────────────────────────────────────────────

@app.route("/gemini_translate", methods=["POST"])
def gemini_translate():
    data     = request.get_json()
    sentence = data.get("sentence", [])
    if not sentence:
        return jsonify({"response": "No signs detected yet. Please sign something first!"})

    sentence_str = " ".join(sentence)
    prompt = f"""You are an Indian Sign Language (ISL) expert integrated into a real-time recognition system.

Detected ISL gestures: {sentence_str}

Please provide:
1. Natural English translation
2. Hindi translation
3. Brief context/explanation

Format:
**Translation:** [English]
**Hindi:** [Hindi]
**Explanation:** [1-2 sentences]"""

    if not GEMINI_AVAILABLE:
        return jsonify({
            "response": (
                f"**Translation:** {sentence_str}\n"
                f"**Hindi:** (हिंदी अनुवाद)\n"
                f"**Explanation:** Set GEMINI_API_KEY env var for real AI translation."
            ),
            "demo_mode": True,
        })
    try:
        resp = gemini_model.generate_content(prompt)
        return jsonify({"response": resp.text, "demo_mode": False})
    except Exception as e:
        return jsonify({"response": f"Gemini error: {e}", "demo_mode": True})


@app.route("/gemini_chat", methods=["POST"])
def gemini_chat():
    data    = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"response": "Please enter a message."})

    prompt = (
        "You are an ISL (Indian Sign Language) expert assistant in a real-time recognition dashboard.\n"
        f"Answer concisely and helpfully: {message}"
    )

    if not GEMINI_AVAILABLE:
        return jsonify({
            "response" : f"Demo mode — set GEMINI_API_KEY. Your question: '{message}'",
            "demo_mode": True,
        })
    try:
        resp = gemini_model.generate_content(prompt)
        return jsonify({"response": resp.text, "demo_mode": False})
    except Exception as e:
        return jsonify({"response": f"Gemini error: {e}", "demo_mode": True})


@app.route("/")
def index():
    return render_template("index1.html")


@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok",
        "mediapipe"   : MEDIAPIPE_OK,
        "gemini"      : GEMINI_AVAILABLE,
        "esp32_live"  : esp32_is_live(),
        "dummy_running": dummy_running,
    })


if __name__ == "__main__":
    print("\n🤟 ISL Sign Language Unified System")
    print("   Webcam + ESP32 Glove + Gemini AI")
    print("   http://localhost:5001")
    print("   Calibration: GET /api/calibrate\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)