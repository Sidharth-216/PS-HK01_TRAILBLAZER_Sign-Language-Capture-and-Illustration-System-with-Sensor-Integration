// ============================================================
// navigation.js — Sidebar & Section Management
// ============================================================

const SECTION_TITLES = {
    dashboard:   'Live Dashboard',
    calibration: 'Sensor Calibration',
    training:    'Training Mode',
    history:     'Gesture History',
    settings:    'Settings',
};

function showSection(id) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));

    // Show target section
    const target = document.getElementById(id);
    if (target) target.classList.add('active');

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === id);
    });

    // Update topbar title
    const titleEl = document.getElementById('topbarTitle');
    if (titleEl) titleEl.textContent = SECTION_TITLES[id] || id;

    // Close sidebar on mobile
    if (window.innerWidth <= 900) {
        document.getElementById('sidebar').classList.remove('open');
    }

    // Trigger section-specific init
    if (id === 'history') loadHistory();
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// Update topbar date
function updateTopbarDate() {
    const el = document.getElementById('topbarDate');
    if (!el) return;
    const now = new Date();
    el.textContent = now.toLocaleDateString('en-IN', {
        weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
    });
}


window.onload = function () {
    showSection('dashboard');
    updateTopbarDate();

    // Simulate connection after 1.5s
    setTimeout(() => updateConnectionStatus(true), 1500);
};