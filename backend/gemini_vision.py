"""
gemini_vision.py  —  Plan B Core Engine
ISL Sign Language Recognition via Google Gemini Vision API

This module:
  1. Opens laptop webcam via OpenCV
  2. Runs MediaPipe Hands locally to detect hand presence (no API cost)
  3. When a hand is detected, encodes the frame as base64 JPEG
  4. Sends it to Gemini 2.5 Flash with a carefully engineered ISL prompt
  5. Returns structured JSON: gesture, confidence, description, alternative
  6. Uses 3-frame majority voting to prevent flickering between similar signs

Usage:
  from gemini_vision import camera_loop, get_latest
  # Run camera_loop() in a background thread
  # Call get_latest() from Flask routes to get current gesture
"""

import cv2
import base64
import json
import os
import time
import threading
from collections import deque, Counter

import google.generativeai as genai
import mediapipe as mp

# ── Gemini Setup ──────────────────────────────────────────────────────────────
# API key must be set as environment variable GEMINI_API_KEY
# Get free key at: aistudio.google.com → Get API Key
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✓ Gemini SDK configured successfully")
except KeyError:
    print("✗ ERROR: GEMINI_API_KEY environment variable not set!")
    print("  Windows: set GEMINI_API_KEY=AIzaSyYourKey")
    print("  Linux/Mac: export GEMINI_API_KEY=AIzaSyYourKey")
    model = None

# ── MediaPipe Hands (runs locally, free, no internet needed) ─────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,       # Video mode for better tracking
    max_num_hands=1,               # We only need one hand
    min_detection_confidence=0.70,
    min_tracking_confidence=0.60,
    model_complexity=1             # 0=fast, 1=balanced, 2=accurate
)

# ── ISL Prompt (most critical part — tune this if signs are misidentified) ────
ISL_PROMPT = """
You are an expert Indian Sign Language (ISL) interpreter with deep knowledge
of hand gestures and finger positions.

Analyze the hand gesture in this image very carefully. Look at:
1. Which fingers are FULLY EXTENDED (pointing straight) vs BENT/CURLED
2. Palm orientation — facing the camera, sideways (left/right), or facing down
3. Thumb position — extended outward, tucked in, resting on fingers
4. The overall shape formed by all five fingers together

FINGER REFERENCE:
- Thumb: the wide short finger on the left side of a right hand
- Index: first tall finger next to thumb
- Middle: tallest finger in center
- Ring: fourth finger
- Pinky: smallest finger on the right of a right hand

Identify the ISL sign from these options:
ALPHABETS: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
WORDS: Hello ThankYou Yes No Please Sorry Help Water Food Good Bad More
       Stop Come Go Love Friend Home School Name What Where How Understand
       NotUnderstand Again Father Mother

IMPORTANT DISAMBIGUATION:
- A: closed fist, thumb rests on the SIDE of the index finger
- S: closed fist, thumb crosses OVER the bent fingers
- E: all fingers bent to touch thumb, making a claw shape
- Hello: open flat palm, all fingers extended, facing camera
- Stop: open palm facing outward like a "halt" sign
- Yes: closed fist nodding / shaking (if motion visible, otherwise fist = A or S)

Respond ONLY with this exact JSON structure, nothing else before or after:
{
  "gesture": "the recognized ISL label exactly as listed",
  "confidence": 0.92,
  "description": "brief 1-sentence description of hand shape observed",
  "is_clear": true,
  "alternative": "second most likely sign if confidence is below 0.80"
}

If no hand is clearly visible, lighting is too dark, or gesture is too ambiguous:
{"gesture": "UNCLEAR", "confidence": 0.0, "description": "no clear hand gesture visible", "is_clear": false, "alternative": ""}
"""

# ── Shared State (thread-safe) ────────────────────────────────────────────────
_state = {
    "gesture":     "—",
    "confidence":  0.0,
    "description": "Waiting for hand gesture...",
    "is_clear":    False,
    "alternative": "",
    "timestamp":   0.0,
    "hand_visible": False,
    "api_calls_today": 0,
    "last_error":  "",
}
_vote_window  = deque(maxlen=3)   # majority vote across last 3 API responses
_last_api_ts  = 0.0
_lock         = threading.Lock()

# ── Config ────────────────────────────────────────────────────────────────────
API_COOLDOWN_SEC   = 1.2    # minimum seconds between Gemini API calls
JPEG_QUALITY       = 75     # lower = faster/cheaper, higher = more accurate
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
CONFIDENCE_MIN     = 0.65   # discard results below this threshold


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def frame_to_b64(frame: "np.ndarray") -> str:
    """Resize frame and encode as base64 JPEG string for Gemini API."""
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buf = cv2.imencode(".jpg", resized, encode_params)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def call_gemini(frame_b64: str) -> dict | None:
    """
    Send frame to Gemini Vision API and parse ISL JSON response.
    Returns parsed dict or None on error.
    """
    if model is None:
        return None
    try:
        response = model.generate_content([
            ISL_PROMPT,
            {"mime_type": "image/jpeg", "data": frame_b64}
        ])
        text = response.text.strip()
        # Strip markdown code fences that Gemini sometimes adds
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError as e:
        print(f"  Gemini JSON parse error: {e} | Raw: {text[:120]}")
        return None
    except Exception as e:
        print(f"  Gemini API error: {e}")
        with _lock:
            _state["last_error"] = str(e)
        return None


def _process_frame_async(b64: str):
    """Called in background thread. Calls Gemini and updates shared state."""
    global _state

    parsed = call_gemini(b64)
    if not parsed:
        return

    # Discard UNCLEAR or low-confidence results
    if not parsed.get("is_clear") or parsed.get("confidence", 0) < CONFIDENCE_MIN:
        with _lock:
            _state["hand_visible"] = True   # hand is there, just not clear enough
        return

    # Majority voting: only update if same gesture appears 2+ times in last 3
    _vote_window.append(parsed["gesture"])
    vote_counts = Counter(_vote_window)
    stable_gesture, vote_count = vote_counts.most_common(1)[0]

    with _lock:
        _state.update({
            "gesture":      stable_gesture,
            "confidence":   round(parsed.get("confidence", 0), 3),
            "description":  parsed.get("description", ""),
            "is_clear":     True,
            "alternative":  parsed.get("alternative", ""),
            "timestamp":    time.time(),
            "hand_visible": True,
            "api_calls_today": _state["api_calls_today"] + 1,
            "last_error":   "",
        })
        print(f"  ✓ Gesture: {stable_gesture:12s} | Conf: {parsed['confidence']*100:.0f}% | Votes: {vote_count}/3")


# ─────────────────────────────────────────────────────────────────────────────
# Main camera loop — run this in a background thread
# ─────────────────────────────────────────────────────────────────────────────

def camera_loop(show_preview: bool = True, camera_index: int = 0):
    """
    Main camera capture loop.
    Call this in a background daemon thread from app.py.

    Args:
        show_preview: If True, opens an OpenCV window with live overlay.
                      Set to False when running headless (Flask server mode).
        camera_index: Webcam index (0 = default laptop camera).
    """
    global _last_api_ts

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"✗ Cannot open camera {camera_index}")
        with _lock:
            _state["last_error"] = f"Cannot open camera {camera_index}"
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    print(f"✓ Camera {camera_index} opened at {FRAME_WIDTH}x{FRAME_HEIGHT}")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("  Camera read failed — retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)                          # Mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── MediaPipe: detect hand landmarks (local, zero API cost) ──────────
        mp_result = hands_detector.process(rgb)
        hand_present = mp_result.multi_hand_landmarks is not None

        if hand_present:
            # Draw hand skeleton on the frame
            for lm in mp_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

            # Rate-limited Gemini API call
            now = time.time()
            if now - _last_api_ts > API_COOLDOWN_SEC:
                _last_api_ts = now
                b64 = frame_to_b64(frame)
                t = threading.Thread(
                    target=_process_frame_async,
                    args=(b64,),
                    daemon=True
                )
                t.start()
        else:
            with _lock:
                _state["hand_visible"] = False

        # ── Optional preview window ───────────────────────────────────────────
        if show_preview:
            with _lock:
                g = _state["gesture"]
                c = _state["confidence"]
                err = _state["last_error"]

            color = (0, 255, 150) if hand_present else (100, 100, 100)
            status = f"{g}  ({c*100:.0f}%)" if hand_present else "Show hand to camera"
            cv2.putText(frame, status, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

            if err:
                cv2.putText(frame, f"ERR: {err[:50]}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

            hand_txt = "HAND DETECTED" if hand_present else "No hand"
            cv2.putText(frame, hand_txt, (20, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 100) if hand_present else (80, 80, 80), 1)

            cv2.imshow("ISL — Gemini Vision (Plan B)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Camera loop stopped by user (Q key)")
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("Camera loop ended.")


def get_latest() -> dict:
    """Return a copy of the current gesture state. Thread-safe."""
    with _lock:
        return dict(_state)


def get_status() -> dict:
    """Return system status for dashboard health check."""
    with _lock:
        return {
            "gemini_configured": model is not None,
            "api_calls_today":   _state["api_calls_today"],
            "last_error":        _state["last_error"],
            "hand_visible":      _state["hand_visible"],
            "cooldown_sec":      API_COOLDOWN_SEC,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — run this file directly to test without Flask
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Gemini Vision — Standalone Test Mode ===")
    print("Show your hand to the camera. Press Q to quit.\n")
    camera_loop(show_preview=True)