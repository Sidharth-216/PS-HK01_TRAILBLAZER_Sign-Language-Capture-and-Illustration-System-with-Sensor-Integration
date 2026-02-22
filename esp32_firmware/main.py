# ============================================================
#  ISL Sign Language Glove — ESP32 MicroPython Firmware
#  File: main.py
#
#  TWO MODES — switch by editing the run() function below:
#
#  MODE 1: LIVE RECOGNITION (default for demo)
#    → PIR detects motion → reads sensors → POSTs to Flask /gesture
#    → Keep raw_srv lines COMMENTED OUT
#
#  MODE 2: DATA COLLECTION (for collect_data.py)
#    → Runs a small HTTP server on port 8080
#    → collect_data.py polls GET /raw to read sensors
#    → Uncomment raw_srv lines in run() to enable
# ============================================================

import machine, network, socket, json, time
from config import WIFI_SSID, WIFI_PASSWORD, SERVER_IP, SERVER_PORT

# ── Status LED (GPIO 2 = built-in LED on most ESP32 boards) ──
led = machine.Pin(2, machine.Pin.OUT)

def blink(n=1, ms=120):
    for _ in range(n):
        led.on();  time.sleep_ms(ms)
        led.off(); time.sleep_ms(ms)

# ── ADC pins for 5 flex sensors ──────────────────────────────
# Wiring: 3.3V → Flex sensor → ADC pin → 10kΩ resistor → GND
thumb_adc  = machine.ADC(machine.Pin(34))
index_adc  = machine.ADC(machine.Pin(35))
middle_adc = machine.ADC(machine.Pin(32))
ring_adc   = machine.ADC(machine.Pin(33))
pinky_adc  = machine.ADC(machine.Pin(26))

SENSORS = {
    "thumb":  thumb_adc,
    "index":  index_adc,
    "middle": middle_adc,
    "ring":   ring_adc,
    "pinky":  pinky_adc,
}

# Full 3.3V range + 12-bit resolution on all channels
for s in SENSORS.values():
    s.atten(machine.ADC.ATTN_11DB)
    s.width(machine.ADC.WIDTH_12BIT)   # values 0-4095

# ── PIR motion sensor (GPIO 27) ───────────────────────────────
pir = machine.Pin(27, machine.Pin.IN)

# ── WiFi ──────────────────────────────────────────────────────
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("Already connected. IP:", wlan.ifconfig()[0])
        return wlan.ifconfig()[0]
    print("Connecting to WiFi:", WIFI_SSID)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    attempts = 0
    while not wlan.isconnected():
        time.sleep(1)
        attempts += 1
        print("  waiting...", attempts)
        if attempts > 25:
            print("WiFi failed. Restarting...")
            machine.reset()
    ip = wlan.ifconfig()[0]
    print("Connected! IP:", ip)
    blink(3, 200)
    return ip

# ── Read sensors (averaged for stability) ────────────────────
def read_sensors(samples=15, delay_ms=8):
    totals = {k: 0 for k in SENSORS}
    for _ in range(samples):
        for name, adc in SENSORS.items():
            totals[name] += adc.read()
        time.sleep_ms(delay_ms)
    return {k: v // samples for k, v in totals.items()}

# ── POST to Flask /gesture (live recognition mode) ───────────
def post_gesture(readings):
    try:
        addr = socket.getaddrinfo(SERVER_IP, SERVER_PORT)[0][-1]
        s = socket.socket()
        s.settimeout(4)
        s.connect(addr)
        body = json.dumps(readings)
        req = (
            "POST /gesture HTTP/1.0\r\n"
            "Host: {}:{}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: {}\r\n"
            "Connection: close\r\n\r\n"
            "{}"
        ).format(SERVER_IP, SERVER_PORT, len(body), body)
        s.send(req.encode())
        resp = s.recv(512).decode()
        s.close()
        return "200" in resp
    except Exception as e:
        print("POST error:", e)
        return False

# ── Raw HTTP server (data-collection mode) ───────────────────
# This serves GET /raw so collect_data.py can read sensor values.
# Only active when raw_srv is enabled in run().

_latest_raw = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}

def start_raw_server(port=8080):
    addr = socket.getaddrinfo("0.0.0.0", port)[0][-1]
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(addr)
    srv.listen(2)
    srv.settimeout(0.05)    # non-blocking
    print("Raw server listening on port", port)
    return srv

def handle_raw_request(srv):
    """Call this every loop iteration when raw server is active."""
    try:
        conn, addr = srv.accept()
        conn.settimeout(1)
        try:
            conn.recv(256)   # read and discard the HTTP request
            body = json.dumps(_latest_raw)
            resp = (
                "HTTP/1.0 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: {}\r\n"
                "Connection: close\r\n\r\n"
                "{}"
            ).format(len(body), body)
            conn.send(resp.encode())
        except Exception:
            pass
        conn.close()
    except OSError:
        pass   # timeout — no client connected, that's fine

# ── MAIN ─────────────────────────────────────────────────────
def run():
    global _latest_raw

    connect_wifi()
    print("Glove ready.")
    blink(5, 80)

    # ── DATA COLLECTION MODE ──────────────────────────────────
    # To collect training data:
    #   1. Uncomment the two lines below
    #   2. Flash to ESP32
    #   3. Run collect_data.py on laptop
    #   4. After collection, re-comment and re-flash for live mode
    #
    # raw_srv = start_raw_server(8080)
    # DATA_COLLECTION = True
    #
    # ── LIVE RECOGNITION MODE (default) ──────────────────────
    raw_srv = None
    DATA_COLLECTION = False

    last_send   = 0
    COOLDOWN_MS = 800       # minimum ms between gesture sends

    while True:
        now = time.ticks_ms()

        # Always read sensors (needed for both modes)
        readings = read_sensors(samples=15)
        _latest_raw = dict(readings)

        if DATA_COLLECTION and raw_srv:
            # ── Collection mode: serve readings over HTTP ────
            handle_raw_request(raw_srv)

        else:
            # ── Live mode: PIR trigger → POST to Flask ───────
            if pir.value() == 1:
                if time.ticks_diff(now, last_send) > COOLDOWN_MS:
                    readings["ts"] = now
                    ok = post_gesture(readings)
                    if ok:
                        led.on(); time.sleep_ms(60); led.off()
                        print("Sent:", readings)
                    else:
                        blink(2, 60)
                    last_send = now

        time.sleep_ms(80)

run()
