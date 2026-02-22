#!/usr/bin/env python3
"""
collect_data.py — Collect real glove sensor data (calibrated, 4-sensor)
Works with the updated main.py that sends NORMALIZED values (0=flat, 4095=bent).
Includes live readout so you can verify sensors are responding correctly.

Run: python ml/collect_data.py
"""

import requests, csv, time, os, sys
import numpy as np

ESP32_IP   = "10.13.124.226"
ESP32_PORT = 8080
URL        = f"http://{ESP32_IP}:{ESP32_PORT}/raw"

SAVE_PATH            = "ml/data/gesture_data.csv"
FEATURE_COLS         = ["thumb", "index", "middle", "ring", "pinky"]
ACTIVE_COLS          = ["thumb", "index", "middle", "ring"]
SAMPLES_PER_GESTURE  = 60

# After calibration, expected ranges:
#   0     = fully flat/straight
#   4095  = fully bent
NORMALIZED_FLAT  = 0
NORMALIZED_BENT  = 4095
NORMALIZED_RANGE = 4095

GESTURES = [
    ("Hello",    "All fingers fully FLAT/STRAIGHT — open palm"),
    ("Yes",      "All fingers fully BENT — tight fist"),
    ("No",       "Index+thumb flat, middle+ring fully bent"),
    ("ThankYou", "All fingers at HALFWAY position (medium bend)"),
    ("Please",   "Thumb bent inward, other fingers only slightly bent"),
    ("Sorry",    "Thumb+middle+ring fully bent, only index pointing up"),
    ("Help",     "Thumb fully flat/straight, all other fingers fully bent"),
    ("Stop",     "Thumb+ring bent, index+middle pointing up straight"),
    ("Good",     "Index bent inward, middle+ring fully flat/straight"),
    ("Bad",      "Index+middle bent, thumb+ring fully flat/straight"),
    ("Come",     "Thumb+middle bent, index+ring straight"),
    ("A",        "Maximum tight fist — all as bent as possible"),
    ("B",        "Thumb bent only, all others fully straight"),
    ("C",        "All fingers in medium C-curve — same bend on all"),
    ("D",        "Index pointing up straight, middle+ring fully bent"),
    ("L",        "Thumb+index straight (L shape), middle+ring fully bent"),
    ("V",        "Thumb+ring bent, index+middle straight up (peace sign)"),
    ("S",        "Thumb over fist — slightly less tight than A"),
    ("F",        "Thumb+index pinched together, middle+ring straight up"),
]

os.makedirs("ml/data", exist_ok=True)


def get_reading():
    r = requests.get(URL, timeout=3)
    r.raise_for_status()
    data = r.json()
    data["pinky"] = 0
    return data


def sensor_bar(val, width=20):
    """Visual bar showing 0 (flat) → full (bent)."""
    filled = int(val / NORMALIZED_RANGE * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def verify_sensors():
    """
    Live sensor readout so user can check calibration is working.
    Shows bars: empty = flat, full = bent.
    """
    print("\n── SENSOR VERIFICATION ─────────────────────────────────")
    print("  Watching live sensor values. Move your fingers and confirm:")
    print("  FLAT finger = bar near empty (0)")
    print("  BENT finger = bar near full (4095)")
    print("  Press Ctrl+C to continue to data collection.\n")
    try:
        while True:
            d = get_reading()
            parts = []
            for f in ACTIVE_COLS:
                v = d.get(f, 0)
                bar = sensor_bar(v)
                parts.append(f"{f[0].upper()}:{bar}{v:4d}")
            print("  " + "  ".join(parts), end="\r")
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n\n  ✓ Verification done — starting collection\n")


def test_connection():
    print(f"\nConnecting to ESP32 at {URL} …")
    try:
        d = get_reading()
        print(f"✓ Connected!")
        print(f"  Raw values: T:{d['thumb']}  I:{d['index']}  M:{d['middle']}  R:{d['ring']}")
        # Quick sanity check — with new firmware, values should be in 0–4095
        # and NOT all near 4095 simultaneously
        high = sum(1 for f in ["index","middle","ring"] if d.get(f,0) > 3800)
        if high >= 2:
            print("\n  ⚠  WARNING: index/middle/ring all reading near 4095.")
            print("     This means ESP32 is running OLD firmware (not calibrated).")
            print("     Please flash the updated main.py first!")
            print("     Data collection will continue but values may be inverted.\n")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\nFix checklist:")
        print("  1. ESP32 powered on + WiFi connected (check Thonny serial)")
        print("  2. main.py is in DATA COLLECTION mode (raw_srv uncommented)")
        print("  3. ESP32_IP is correct")
        return False


def quality_score(readings):
    if len(readings) < 3: return 100
    arr  = np.array(readings)
    stds = arr.std(axis=0)
    score = 100
    if stds.max() > 400: score -= 20   # too shaky
    if stds.max() < 5:   score -= 30   # sensor frozen?
    return max(0, score)


def collect_gesture(gesture_name, description, writer, n_samples=SAMPLES_PER_GESTURE):
    print(f"\n{'─'*55}")
    print(f"  GESTURE : {gesture_name}")
    print(f"  HOW TO  : {description}")
    print(f"{'─'*55}")
    input(f"  → Hold the sign firmly, then press ENTER to start… ")
    print(f"  HOLD STILL — capturing {n_samples} samples… (Ctrl+C to skip)\n")

    readings = []
    errors   = 0

    while len(readings) < n_samples:
        try:
            data = get_reading()
            readings.append([data.get(k, 0) for k in ACTIVE_COLS])
            writer.writerow([data.get(k, 0) for k in FEATURE_COLS] + [gesture_name])

            pct = len(readings) / n_samples * 100
            bar = "█" * (len(readings) // (n_samples // 20))
            vals = "  ".join(f"{f[0].upper()}:{data.get(f,0):4d}" for f in ACTIVE_COLS)
            print(f"  [{len(readings):2d}/{n_samples}] {vals}  [{bar:<20}] {pct:.0f}%", end="\r")
            time.sleep(0.12)

        except KeyboardInterrupt:
            print(f"\n\n  ⏭  Skipped '{gesture_name}'")
            return 0
        except Exception as e:
            errors += 1
            if errors > 8:
                print(f"\n  ✗ Too many errors — skipping.")
                return 0
            time.sleep(0.8)

    print()
    q = quality_score(readings)
    status = "✓" if q >= 70 else "⚠"
    print(f"  {status}  {gesture_name}: {len(readings)} samples (quality: {q}/100)")
    if q < 70:
        print(f"     Low quality — try re-collecting this gesture.")
    if np.array(readings).std(axis=0).max() < 5:
        print(f"     ⚠ Very low variance — sensor may be stuck or sign too rigid.")
    return len(readings)


def main():
    print("=" * 55)
    print("  ISL Glove — Data Collection (calibrated sensors)")
    print("=" * 55)

    if not test_connection():
        sys.exit(1)

    print(f"\n  Samples per gesture : {SAMPLES_PER_GESTURE}")
    print(f"  Total gestures      : {len(GESTURES)}")
    print(f"  Estimated time      : ~{len(GESTURES) * SAMPLES_PER_GESTURE * 0.15 / 60:.0f} min")
    print(f"  Save path           : {SAVE_PATH}")
    print(f"\n  TIP: Hold each sign COMPLETELY STILL while recording.")
    print(f"  TIP: 0 = flat/straight, 4095 = fully bent (after calibration)\n")

    # Offer live sensor check
    check = input("  Run live sensor verification first? (y/n): ").strip().lower()
    if check == "y":
        verify_sensors()

    print("\nGesture list:")
    for i, (g, desc) in enumerate(GESTURES):
        print(f"  {i+1:2d}. {g:<12} {desc}")

    print("\nOptions:")
    choice = input("  Enter 1 for ALL, or numbers e.g. '2 5 7': ").strip()

    if choice == "1" or choice == "":
        to_collect = GESTURES
    else:
        try:
            idxs = [int(x)-1 for x in choice.split()]
            to_collect = [GESTURES[i] for i in idxs if 0 <= i < len(GESTURES)]
        except Exception:
            print("  Bad input — collecting all")
            to_collect = GESTURES

    write_header = not os.path.exists(SAVE_PATH)
    total_saved  = 0

    with open(SAVE_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FEATURE_COLS + ["label"])

        for gesture_name, description in to_collect:
            saved = collect_gesture(gesture_name, description, writer)
            total_saved += saved

    print(f"\n{'='*55}")
    print(f"  Done! {total_saved} total samples saved.")
    print(f"  File: {SAVE_PATH}")
    print(f"\n  Retrain the model:")
    print(f"    python ml/train_model.py --no-download")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()