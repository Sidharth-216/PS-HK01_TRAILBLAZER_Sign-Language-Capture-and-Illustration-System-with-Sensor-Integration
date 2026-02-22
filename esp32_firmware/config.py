# ============================================================
#  config.py — WiFi & Server Configuration
#  EDIT THIS FILE before flashing to ESP32
# ============================================================

# ── Your WiFi credentials (use mobile hotspot during hackathon)
WIFI_SSID     = "Sidhu"
WIFI_PASSWORD = "12345678"

# ── IP of the laptop running Flask
# Run `ipconfig` (Windows) or `ifconfig` (Linux/Mac) to find it
SERVER_IP   = "10.251.253.56"   # change this!
SERVER_PORT = 5000

# ── Sensor calibration (raw ADC values 0-4095)
# Measure these with your actual glove and update:
CALIBRATION = {
    "thumb":  {"flat": 1800, "bent": 3200},
    "index":  {"flat": 1750, "bent": 3100},
    "middle": {"flat": 1800, "bent": 3150},
    "ring":   {"flat": 1700, "bent": 3000},
    "pinky":  {"flat": 1650, "bent": 2900},
}
