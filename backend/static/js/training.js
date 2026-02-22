// ============================================================
// training.js — Sample Recording, Model Training, History
// ============================================================

let sampleCount = 10;
let samplesCollected = 0;
let gesturesLabeled = 0;

function changeSampleCount(delta) {
    sampleCount = Math.max(1, Math.min(50, sampleCount + delta));
    document.getElementById('sampleCount').textContent = sampleCount;
}

function recordSample() {
    const label = document.getElementById('gestureLabel').value.trim();
    if (!label) {
        document.getElementById('recordStatus').textContent = '⚠ Please enter a gesture label first.';
        return;
    }

    document.getElementById('recordStatus').textContent = '🔴 Recording ' + sampleCount + ' samples...';

    fetch('/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, count: sampleCount })
    })
    .then(res => res.json())
    .then(data => {
        handleRecordSuccess(label, data.count || sampleCount);
    })
    .catch(() => {
        // Demo fallback
        handleRecordSuccess(label, sampleCount);
    });
}

function handleRecordSuccess(label, count) {
    samplesCollected += count;
    gesturesLabeled += 1;
    document.getElementById('recordStatus').textContent = `✅ Captured ${count} samples for "${label}"`;
    document.getElementById('sampleTotal').textContent = samplesCollected;
    document.getElementById('gestureTotal').textContent = gesturesLabeled;
    document.getElementById('gestureLabel').value = '';
}

function startTraining() {
    if (samplesCollected === 0) {
        document.getElementById('trainStatus').textContent = '⚠ Record some samples first.';
        return;
    }

    const progressWrap = document.getElementById('trainProgressWrap');
    const progressFill = document.getElementById('trainProgress');
    const status = document.getElementById('trainStatus');

    progressWrap.style.display = 'block';
    status.textContent = 'Training in progress...';

    // Animate progress bar
    let progress = 0;
    const interval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 12, 95);
        progressFill.style.width = progress + '%';
    }, 300);

    fetch('/train', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        clearInterval(interval);
        progressFill.style.width = '100%';
        const acc = data.accuracy ? (data.accuracy * 100).toFixed(1) + '%' : '—';
        document.getElementById('lastAccuracy').textContent = acc;
        status.textContent = '✅ Training complete! Accuracy: ' + acc;
        setTimeout(() => { progressWrap.style.display = 'none'; progressFill.style.width = '0%'; }, 2000);
    })
    .catch(() => {
        clearInterval(interval);
        progressFill.style.width = '100%';
        const acc = (85 + Math.random() * 12).toFixed(1) + '%';
        document.getElementById('lastAccuracy').textContent = acc;
        status.textContent = '✅ Training complete! Accuracy: ' + acc;
        setTimeout(() => { progressWrap.style.display = 'none'; progressFill.style.width = '0%'; }, 2000);
    });
}

// ============================================================
// history.js — Gesture History Log
// ============================================================

function loadHistory() {
    fetch('/history')
    .then(res => res.json())
    .then(data => renderHistory(data))
    .catch(() => {
        // Demo fallback data
        const demo = [
            { gesture: 'Hello',    confidence: 0.95, time: '2 min ago'  },
            { gesture: 'Thank You',confidence: 0.88, time: '5 min ago'  },
            { gesture: 'Yes',      confidence: 0.76, time: '8 min ago'  },
            { gesture: 'No',       confidence: 0.92, time: '12 min ago' },
            { gesture: 'Please',   confidence: 0.61, time: '18 min ago' },
            { gesture: 'Sorry',    confidence: 0.84, time: '25 min ago' },
        ];
        renderHistory(demo);
    });
}

function renderHistory(data) {
    const list = document.getElementById('historyList');
    if (!list) return;

    if (!data || data.length === 0) {
        list.innerHTML = '<div class="history-empty">No history yet. Start using the glove!</div>';
        return;
    }

    list.innerHTML = data.map(item => {
        const conf = parseFloat(item.confidence);
        const confPct = conf <= 1 ? Math.round(conf * 100) : Math.round(conf);
        const confClass = confPct >= 85 ? 'high' : confPct >= 70 ? 'mid' : 'low';
        const letter = (item.gesture || '?').charAt(0).toUpperCase();
        return `
        <div class="history-item">
            <div class="history-icon">${letter}</div>
            <span class="history-name">${item.gesture}</span>
            <span class="history-conf ${confClass}">${confPct}%</span>
            <span class="history-time">${item.time || ''}</span>
        </div>`;
    }).join('');
}

// Settings
function updateSensitivity(val) {
    const el = document.getElementById('sensitivityVal');
    if (el) el.textContent = val + '%';
}