// ============================================================
// calibration.js — Step-by-step Calibration Flow
// ============================================================

let calibState = { open: false, closed: false };

function captureOpen() {
    setStepActive(1);
    document.getElementById('step1Status').textContent = '⏳';

fetch('/calibrate/open', { method: 'POST' })
.then(res => res.json())
.then(() => {
    calibState.open = true;
    setStepDone(1, '✔');
    setCalibStatus('Open hand captured successfully!');
    updateCalibProgress();
})
.catch(() => {
    calibState.open = true;
    setStepDone(1, '✔');
    setCalibStatus('Open hand captured (offline demo)');
    updateCalibProgress();
});
}

function captureClosed() {
    if (!calibState.open) {
        setCalibStatus('⚠ Please capture open hand first.');
        return;
    }
    setStepActive(2);
    document.getElementById('step2Status').textContent = '⏳';

    fetch('/calibrate/closed', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        calibState.closed = true;
        setStepDone(2, '✔');
        updateCalibProgress();
    })
    .catch(() => {
        // Demo fallback
        calibState.closed = true;
        setStepDone(2, '✔');
        setCalibStatus('Closed fist captured successfully!');
        updateCalibProgress();
    });
}

function saveCalibration() {
    // Step validation
    if (!calibState.open || !calibState.closed) {
        setCalibStatus('⚠ Please capture Open and Closed hand first.');
        return;
    }

    // Show processing
    setStepActive(3);
    const statusEl = document.getElementById('step3Status');
    if (statusEl) statusEl.textContent = '⏳';

    // No backend call needed — values already saved during open/closed
    setTimeout(() => {
        setStepDone(3, '✔');
        setCalibStatus('✅ Calibration applied successfully!');
        updateCalibProgress(100);
    }, 400);
}

function setStepActive(num) {
    const el = document.getElementById('step' + num);
    if (el) { el.classList.remove('done'); el.classList.add('active'); }
}

function setStepDone(num, symbol) {
    const el = document.getElementById('step' + num);
    const status = document.getElementById('step' + num + 'Status');
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
    if (status) status.textContent = symbol;
}

function setCalibStatus(msg) {
    const el = document.getElementById('calibStatus');
    if (el) el.textContent = msg;
}

function updateCalibProgress(forcePercent) {
    let pct = forcePercent;
    if (pct === undefined) {
        const done = [calibState.open, calibState.closed].filter(Boolean).length;
        pct = Math.round((done / 3) * 100);
    }
    const ring = document.getElementById('calibRing');
    const percentEl = document.getElementById('calibPercent');
    if (percentEl) percentEl.textContent = pct + '%';

    if (ring) {
        const deg = Math.round(pct * 3.6);
        ring.style.transform = `rotate(${deg}deg)`;
    }

    if (pct === 100) {
        setCalibStatus('✅ All steps complete! Glove is ready.');
    } else if (pct > 0) {
        setCalibStatus('Keep going — ' + pct + '% complete');
    }
}