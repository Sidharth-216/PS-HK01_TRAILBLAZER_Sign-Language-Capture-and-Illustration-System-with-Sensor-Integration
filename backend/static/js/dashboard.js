// ============================================================
// dashboard.js — Live Prediction, Sensor Bars, Sentence Builder
// ============================================================

let lastGesture = '';
let sentence = [];
let lastDataTime = Date.now();

function sendDummyData() {
    const sensors = {
        thumb: Math.floor(Math.random() * 2500 + 500),
        index: Math.floor(Math.random() * 2500 + 500),
        middle: Math.floor(Math.random() * 2500 + 500),
        ring: Math.floor(Math.random() * 2500 + 500),
        pinky: Math.floor(Math.random() * 2500 + 500),
    };

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sensors)
    })
        .then(res => res.json())
        .then(data => {
            updateGestureDisplay(data.gesture || 'Hello', data.confidence || '0.87');
            animateSensorBars(sensors);
        })
        .catch(() => {
            // Demo fallback when backend not connected
            const gestures = [
                { letter: 'A', name: 'Hello' },
                { letter: 'B', name: 'Thank You' },
                { letter: 'C', name: 'Yes' },
                { letter: 'D', name: 'No' },
                { letter: 'E', name: 'Please' },
            ];
            const pick = gestures[Math.floor(Math.random() * gestures.length)];
            const conf = (0.75 + Math.random() * 0.24).toFixed(2);
            updateGestureDisplay(pick.name, conf, pick.letter);
            animateSensorBars(sensors);
        });
}

function updateGestureDisplay(gesture, confidenceRaw, letter) {
    lastGesture = gesture;

    const conf = parseFloat(confidenceRaw);
    const confPct = Math.round(conf * 100);

    // Letter display
    const letterEl = document.getElementById('gesture');
    const nameEl = document.getElementById('gestureName');
    if (letterEl) {
        letterEl.textContent = letter || gesture.charAt(0).toUpperCase();
        letterEl.style.transform = 'scale(1.15)';
        setTimeout(() => { letterEl.style.transform = 'scale(1)'; }, 200);
    }
    if (nameEl) nameEl.textContent = gesture;

    // Confidence bar
    const confEl = document.getElementById('confidence');
    const barEl = document.getElementById('confBar');
    if (confEl) confEl.textContent = confPct + '%';
    if (barEl) barEl.style.width = confPct + '%';
}

function animateSensorBars(sensors) {
    const max = 3000;
    const keys = ['thumb', 'index', 'middle', 'ring', 'pinky'];

    keys.forEach(key => {
        const v = sensors[key] || 0;
        const pct = Math.min(Math.round((v / max) * 100), 100);

        // ===== Existing UI (bars + values) =====
        const bar = document.getElementById('bar-' + key);
        const val = document.getElementById('val-' + key);
        if (bar) bar.style.height = pct + '%';
        if (val) val.textContent = v;

    });
    if (window.updateHand) {
        updateHand(sensors);
    }
}

// Sentence Builder
function addWordToSentence() {
    if (!lastGesture) return;
    sentence.push(lastGesture);
    renderSentence();
}

function clearSentence() {
    sentence = [];
    renderSentence();
}

function renderSentence() {
    const el = document.getElementById('sentenceDisplay');
    if (!el) return;
    if (sentence.length === 0) {
        el.innerHTML = '<span class="sentence-placeholder">Detected words will appear here...</span>';
        return;
    }
    el.innerHTML = sentence.map(w =>
        `<span class="sentence-word">${w}</span>`
    ).join('');
}

function speakSentence() {
    if (sentence.length === 0) return;
    const text = sentence.join(' ');
    if ('speechSynthesis' in window) {
        const utt = new SpeechSynthesisUtterance(text);
        utt.lang = 'en-IN';
        window.speechSynthesis.speak(utt);
    }
}
function fetchLiveData() {
    fetch('/latest')
        .then(res => res.json())
        .then(data => {
            if (!data.sensors) return;

            lastDataTime = Date.now();
            updateConnectionStatus(true);

            updateGestureDisplay(data.gesture, data.confidence);
            animateSensorBars(data.sensors);
        })
        .catch(() => { });
}

function updateConnectionStatus(connected) {
    const pill = document.getElementById('connectionStatus');
    if (!pill) return;

    if (connected) {
        pill.classList.add('connected');
        pill.querySelector('.status-text').textContent = 'CONNECTED';
    } else {
        pill.classList.remove('connected');
        pill.querySelector('.status-text').textContent = 'DISCONNECTED';
    }
}

// Check timeout every second
setInterval(() => {
    const now = Date.now();
    if (now - lastDataTime > 2000) {
        updateConnectionStatus(false);
    }

    // Stop motion when disconnected
    if (window.updateHand) {
        updateHand({ thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 });
    }
}, 1000);

// ---- Simulation mode when ESP not connected ----
setInterval(() => {
    if (Date.now() - lastDataTime > 2000) {
        sendDummyData();
    }
}, 200);

// Poll every 100 ms
setInterval(fetchLiveData, 100);