# boot.py — runs before main.py on every power-on
# Keep minimal — just disable debug output

import esp
esp.osdebug(None)   # silence noisy boot logs

import gc
gc.collect()        # free memory before main starts

import webrepl
webrepl.start()     # optional: allows OTA file updates over WiFi
