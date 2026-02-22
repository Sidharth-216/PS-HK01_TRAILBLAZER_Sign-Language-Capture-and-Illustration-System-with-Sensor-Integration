// ============================================================
// hand.js  —  Ultra-Realistic Procedural Hand  (Three.js r128)
// No external models. Full sensor-driven animation.
// ============================================================
'use strict';

/* ------------------------------------------------------------------ */
/*  GLOBALS                                                             */
/* ------------------------------------------------------------------ */
let _scene, _camera, _renderer, _composer;
let _handRoot, _wristBone;
let _shadowMesh, _glowMesh;
let _clock;
let _idleT = 0;
let _scanLineT = 0;
let _lastSensors = { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };

// Exposed so dashboard.js can read joint refs if needed
window.fingers = {};

/* Geometry / material cache — built ONCE at init */
const _M = {};   // materials
const _G = {};   // geometries (shared)

/* ------------------------------------------------------------------ */
/*  SKIN COLOR PALETTE                                                  */
/* ------------------------------------------------------------------ */
const SKIN = {
    palm: 0xd4956a,
    finger: 0xc87c52,
    knuckle: 0xbe7048,
    tip: 0xa85c38,
    nail: 0xf4c8b0,
    vein: 0x9b6040,
};

/* ------------------------------------------------------------------ */
/*  MATERIAL FACTORY                                                    */
/* ------------------------------------------------------------------ */
function buildMaterials() {

    // Palm — warmer, slightly lighter
    _M.palm = new THREE.MeshStandardMaterial({
        color: SKIN.palm,
        roughness: 0.78,
        metalness: 0.0,
        envMapIntensity: 0.3,
    });

    // Finger body — medium tone
    _M.finger = new THREE.MeshStandardMaterial({
        color: SKIN.finger,
        roughness: 0.72,
        metalness: 0.0,
    });

    // Knuckle highlight — darker, slightly glossier
    _M.knuckle = new THREE.MeshStandardMaterial({
        color: SKIN.knuckle,
        roughness: 0.55,
        metalness: 0.02,
    });

    // Fingertip — darkest
    _M.tip = new THREE.MeshStandardMaterial({
        color: SKIN.tip,
        roughness: 0.60,
        metalness: 0.0,
    });

    // Nail — translucent-pink gloss
    _M.nail = new THREE.MeshStandardMaterial({
        color: SKIN.nail,
        roughness: 0.18,
        metalness: 0.08,
        transparent: true,
        opacity: 0.88,
    });

    // Scan-ring glow (emissive ring around wrist)
    _M.scanRing = new THREE.MeshBasicMaterial({
        color: 0x00e5ff,
        transparent: true,
        opacity: 0.55,
        side: THREE.FrontSide,
        depthWrite: false,
    });

    // Soft shadow disc
    _M.shadow = new THREE.MeshBasicMaterial({
        color: 0x000020,
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
    });

    // Ambient glow sprite behind hand
    _M.glow = new THREE.MeshBasicMaterial({
        color: 0x006090,
        transparent: true,
        opacity: 0.10,
        side: THREE.DoubleSide,
        depthWrite: false,
    });
}

/* ------------------------------------------------------------------ */
/*  SHARED GEOMETRY FACTORY (avoids repeated new calls)                 */
/* ------------------------------------------------------------------ */
function buildGeometries() {
    _G.sphere8 = new THREE.SphereGeometry(1, 18, 12);
    _G.sphere16 = new THREE.SphereGeometry(1, 22, 16);
    _G.disc = new THREE.CircleGeometry(1, 48);
    _G.ring = new THREE.TorusGeometry(1, 0.08, 16, 60);
    _G.plane = new THREE.PlaneGeometry(1, 1);
}

/* ------------------------------------------------------------------ */
/*  LIGHTING  —  4-light cinematic rig + point fill                     */
/* ------------------------------------------------------------------ */
function setupLighting() {

    // Sky dome — warm daylight
    const hemi = new THREE.HemisphereLight(0xfff4e8, 0x8090b0, 0.65);
    _scene.add(hemi);

    // Key — strong warm from upper-front-right, casts soft shadows
    const key = new THREE.DirectionalLight(0xfff8f0, 1.6);
    key.position.set(7, 14, 9);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 60;
    key.shadow.camera.left = key.shadow.camera.bottom = -12;
    key.shadow.camera.right = key.shadow.camera.top = 12;
    key.shadow.bias = -0.0015;
    key.shadow.radius = 3;    // PCFSoft blur
    _scene.add(key);

    // Fill — cool blue from lower-left
    const fill = new THREE.DirectionalLight(0xc8e0ff, 0.40);
    fill.position.set(-9, 2, 6);
    _scene.add(fill);

    // Rim — warm orange backlight for skin translucency illusion
    const rim = new THREE.PointLight(0xff9060, 1.0, 30);
    rim.position.set(0, -5, -8);
    _scene.add(rim);

    // Bounce — subtle light from below (table reflection)
    const bounce = new THREE.DirectionalLight(0xffe8d0, 0.20);
    bounce.position.set(0, -10, 5);
    _scene.add(bounce);
}

/* ------------------------------------------------------------------ */
/*  PALM  —  multi-mesh compound shape                                  */
/* ------------------------------------------------------------------ */
function createPalm() {
    const g = new THREE.Group();
    g.name = 'palm';

    // ── Lower palm slab ──
    const lpGeo = new THREE.BoxGeometry(5.8, 1.05, 2.8, 5, 2, 5);
    _sculptBox(lpGeo, 0.12);   // slight vertex noise for organic feel
    const lp = new THREE.Mesh(lpGeo, _M.palm);
    lp.position.set(0, 0, -0.6);
    lp.castShadow = lp.receiveShadow = true;
    g.add(lp);

    // ── Upper palm (toward fingers) ──
    const upGeo = new THREE.BoxGeometry(5.6, 0.90, 2.4, 5, 2, 4);
    _sculptBox(upGeo, 0.08);
    const up = new THREE.Mesh(upGeo, _M.palm);
    up.position.set(0, 0.05, 1.3);
    up.castShadow = up.receiveShadow = true;
    g.add(up);

    // ── Wrist cylinder ──
    const wrGeo = new THREE.CylinderGeometry(1.55, 1.95, 1.6, 24, 3);
    const wr = new THREE.Mesh(wrGeo, _M.palm);
    wr.position.set(0.15, -0.08, -2.6);
    wr.castShadow = wr.receiveShadow = true;
    g.add(wr);
    _wristBone = wr;

    // ── Wrist end cap ──
    const wcGeo = _G.sphere8.clone();
    wcGeo.scale(1.55, 0.8, 1.55);
    const wc = new THREE.Mesh(wcGeo, _M.palm);
    wc.position.set(0.15, -0.08, -3.35);
    wc.rotation.x = Math.PI / 2;
    wc.castShadow = true;
    g.add(wc);

    // ── Thumb mount mound ──
    const tmGeo = _G.sphere8.clone();
    tmGeo.scale(1.1, 0.7, 0.9);
    const tm = new THREE.Mesh(tmGeo, _M.palm);
    tm.position.set(-2.6, 0.15, 0.2);
    tm.castShadow = true;
    g.add(tm);

    // ── Pinky mount mound ──
    const pmGeo = _G.sphere8.clone();
    pmGeo.scale(0.85, 0.65, 0.75);
    const pm = new THREE.Mesh(pmGeo, _M.palm);
    pm.position.set(2.5, 0.12, 1.0);
    pm.castShadow = true;
    g.add(pm);

    // ── 5 knuckle bumps ──
    const kxList = [-2.20, -1.10, 0.02, 1.14, 2.24];
    kxList.forEach((kx, i) => {
        const r = [0.44, 0.46, 0.48, 0.45, 0.40][i];
        const kGeo = _G.sphere8.clone();
        kGeo.scale(r, r * 0.85, r);
        const km = new THREE.Mesh(kGeo, _M.knuckle);
        km.position.set(kx, 0.40, 2.10);
        km.castShadow = true;
        g.add(km);
    });

    _handRoot.add(g);
    return g;
}

/* ─── tiny helper: push surface verts for organic look ─── */
function _sculptBox(geo, strength) {
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const y = pos.getY(i);
        // Arch the top surface gently
        const arch = Math.cos(pos.getX(i) / 3.5) * 0.08 * strength * 8;
        pos.setY(i, y + (Math.random() - 0.5) * strength + (y > 0 ? arch : 0));
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
}

/* ------------------------------------------------------------------ */
/*  FINGER SEGMENT                                                      */
/* ------------------------------------------------------------------ */
/*
  Returns a Group containing:
    - a tapered CylinderGeometry for the bone
    - a joint sphere at the proximal end
    - (optionally) a rounded tip sphere + nail at the distal end
*/
function makeSegment(cfg) {
    const { rTop, rBot, len, mat, isDistal } = cfg;
    const g = new THREE.Group();

    // Cylinder body
    const cGeo = new THREE.CylinderGeometry(rTop, rBot, len, 20, 2);
    const c = new THREE.Mesh(cGeo, mat);
    c.position.y = len * 0.5;
    c.castShadow = c.receiveShadow = true;
    g.add(c);

    // Knuckle sphere at base
    const jGeo = _G.sphere8.clone();
    const jr = rBot * 1.12;
    jGeo.scale(jr, jr * 0.92, jr * 1.05);
    const j = new THREE.Mesh(jGeo, _M.knuckle);
    j.position.y = 0;
    j.castShadow = true;
    g.add(j);

    // Joint sphere at top
    const jtGeo = _G.sphere8.clone();
    const jtr = rTop * 1.08;
    jtGeo.scale(jtr, jtr * 0.90, jtr);
    const jt = new THREE.Mesh(jtGeo, _M.knuckle);
    jt.position.y = len;
    jt.castShadow = true;
    g.add(jt);

    if (isDistal) {
        // Rounded fingertip pad
        const tpGeo = _G.sphere16.clone();
        tpGeo.scale(rTop * 1.22, rTop * 1.35, rTop * 1.10);
        const tp = new THREE.Mesh(tpGeo, _M.tip);
        tp.position.y = len + rTop * 0.85;
        tp.castShadow = true;
        g.add(tp);

        // Fingernail plate
        const nGeo = new THREE.BoxGeometry(rTop * 1.5, rTop * 0.12, rTop * 1.3);
        const nail = new THREE.Mesh(nGeo, _M.nail);
        nail.position.set(0, len + rTop * 1.05, -rTop * 0.35);
        nail.rotation.x = -0.28;
        nail.castShadow = false;
        g.add(nail);
    }

    return g;
}

/* ------------------------------------------------------------------ */
/*  CREATE ONE FINGER                                                   */
/* ------------------------------------------------------------------ */
/*
  cfg: { name, ox, oz, lens[3], radii[4], spread, isThumb }
*/
function createFinger(cfg) {
    const { name, ox, oz, lens, radii, spread, isThumb } = cfg;
    const mat = _M.finger;

    // ── j1 root group at base of finger ──
    const j1 = new THREE.Group();
    j1.name = name + '_j1';
    j1.position.set(ox, 0.42, oz);
    if (isThumb) {
        j1.rotation.z = -0.52;
        j1.rotation.y = -0.28;
    } else {
        j1.rotation.y = spread;
    }

    const seg1 = makeSegment({ rTop: radii[1], rBot: radii[0], len: lens[0], mat, isDistal: false });
    j1.add(seg1);

    // ── j2 hangs off top of seg1 ──
    const j2 = new THREE.Group();
    j2.name = name + '_j2';
    j2.position.y = lens[0];
    seg1.add(j2);

    const seg2 = makeSegment({ rTop: radii[2], rBot: radii[1], len: lens[1], mat, isDistal: false });
    j2.add(seg2);

    // ── j3 hangs off top of seg2 ──
    const j3 = new THREE.Group();
    j3.name = name + '_j3';
    j3.position.y = lens[1];
    seg2.add(j3);

    const seg3 = makeSegment({ rTop: radii[3], rBot: radii[2], len: lens[2], mat, isDistal: true });
    j3.add(seg3);

    _handRoot.add(j1);

    window.fingers[name] = {
        j1, j2, j3,
        isThumb,
        naturalSpread: spread || 0,
        angles: { j1x: 0, j2x: 0, j3x: 0, j1y: isThumb ? -0.28 : (spread || 0) },
    };
}

/* ------------------------------------------------------------------ */
/*  FINGER LAYOUT                                                       */
/* ------------------------------------------------------------------ */
function buildFingers() {
    const defs = [
        { name: 'thumb', ox: -2.55, oz: 0.70, lens: [1.45, 1.25, 1.05], radii: [0.46, 0.41, 0.34, 0.27], spread: 0, isThumb: true },
        { name: 'index', ox: -1.58, oz: 2.18, lens: [1.75, 1.40, 1.12], radii: [0.42, 0.37, 0.31, 0.24], spread: 0.065, isThumb: false },
        { name: 'middle', ox: -0.35, oz: 2.34, lens: [1.90, 1.50, 1.22], radii: [0.44, 0.38, 0.32, 0.25], spread: 0.008, isThumb: false },
        { name: 'ring', ox: 0.88, oz: 2.22, lens: [1.72, 1.40, 1.12], radii: [0.40, 0.36, 0.30, 0.23], spread: -0.052, isThumb: false },
        { name: 'pinky', ox: 2.02, oz: 1.92, lens: [1.35, 1.12, 0.92], radii: [0.34, 0.30, 0.25, 0.19], spread: -0.125, isThumb: false },
    ];
    defs.forEach(d => createFinger(d));
}

/* ------------------------------------------------------------------ */
/*  DECORATIVE ELEMENTS                                                 */
/* ------------------------------------------------------------------ */
function createSceneDressing() {

    // Soft ellipse shadow using scaled circle
    const sdGeo = new THREE.CircleGeometry(1, 48);
    sdGeo.scale(4.8, 2.6, 1);

    _shadowMesh = new THREE.Mesh(sdGeo, _M.shadow);
    _shadowMesh.rotation.x = -Math.PI / 2;
    _shadowMesh.position.set(0.2, -2.35, 0);
    _shadowMesh.renderOrder = -1;
    _scene.add(_shadowMesh);

    // Ambient glow disc behind hand
    const glGeo = _G.disc.clone();
    glGeo.scale(10, 10, 10);
    _glowMesh = new THREE.Mesh(glGeo, _M.glow);
    _glowMesh.position.set(0, 2, -4);
    _glowMesh.renderOrder = -2;
    _scene.add(_glowMesh);

    // Cyan scan ring around wrist
    const srGeo = _G.ring.clone();
    srGeo.scale(2.0, 2.0, 2.0);
    const sr = new THREE.Mesh(srGeo, _M.scanRing);
    sr.name = 'scanRing';
    sr.position.set(0.15, -0.08, -2.6);
    sr.rotation.x = Math.PI / 2;
    _scene.add(sr);

    // Tiny scan line plane (decorative HUD)
    const slGeo = new THREE.PlaneGeometry(12, 0.03);
    const slMat = new THREE.MeshBasicMaterial({
        color: 0x00e5ff, transparent: true, opacity: 0.35,
        depthWrite: false, side: THREE.DoubleSide,
    });
    const sl = new THREE.Mesh(slGeo, slMat);
    sl.name = 'scanLine';
    sl.rotation.x = -Math.PI / 2;
    sl.position.set(0, -2.30, 0);
    _scene.add(sl);
}

/* ------------------------------------------------------------------ */
/*  INIT                                                                */
/* ------------------------------------------------------------------ */
function initHand() {
    const canvas = document.getElementById('handCanvas');
    if (!canvas) { console.warn('handCanvas not found'); return; }

    // Force canvas to fill its CSS container
    const W = canvas.parentElement ? canvas.parentElement.clientWidth : 500;
    const H = canvas.parentElement ? canvas.parentElement.clientHeight : 380;

    // If height is zero, force default
    if (!H || H === 0) H = 380;

    // ── Renderer ──
    _renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    _renderer.setSize(W, H);
    _renderer.shadowMap.enabled = true;
    _renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    _renderer.physicallyCorrectLights = true;
    _renderer.outputEncoding = THREE.sRGBEncoding;
    _renderer.toneMapping = THREE.ACESFilmicToneMapping;
    _renderer.toneMappingExposure = 1.15;

    // ── Scene ──
    _scene = new THREE.Scene();
    _scene.fog = new THREE.FogExp2(0x0a0f1a, 0.038);

    // ── Camera ──
    _camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 200);
    _camera.position.set(3.0, 10.5, 17.5);
    _camera.lookAt(0.3, 2.2, 0);

    _clock = new THREE.Clock();

    // ── Build scene ──
    buildMaterials();
    buildGeometries();
    setupLighting();

    _handRoot = new THREE.Group();
    _handRoot.rotation.x = -0.10;
    _scene.add(_handRoot);

    createPalm();
    buildFingers();
    createSceneDressing();

    window.addEventListener('resize', _onResize);

    _animate();
}

/* ------------------------------------------------------------------ */
/*  UPDATE HAND  —  called by dashboard.js                             */
/* ------------------------------------------------------------------ */
function updateHand(sensors) {
    const MAX = 3000;
    const LF = 0.12;   // lerp factor — lower = smoother/slower

    Object.keys(window.fingers).forEach(name => {
        const f = window.fingers[name];
        let sv = Math.max(0, Math.min(MAX, sensors[name] || 0));
        let t = sv / MAX;

        // Natural coupling
        if (name === 'ring') {
            const m = (sensors.middle || 0) / MAX;
            t = t * 0.85 + m * 0.15;
        }

        if (name === 'pinky') {
            const r = (sensors.ring || 0) / MAX;
            t = t * 0.75 + r * 0.25;
        }

        // Joint angle limits (rad)
        const limA = name === 'thumb' ? 0.68 : 0.82;
        const limB = name === 'thumb' ? 0.60 : 1.08;
        const limC = name === 'thumb' ? 0.52 : 1.25;

        const tA = t * limA;
        const tB = t * limB;
        const tC = t * limC;

        // Lerp
        f.angles.j1x += (tA - f.angles.j1x) * LF;
        f.angles.j2x += (tB - f.angles.j2x) * LF;
        f.angles.j3x += (tC - f.angles.j3x) * LF;

        f.j1.rotation.x = f.angles.j1x;
        f.j2.rotation.x = f.angles.j2x;
        f.j3.rotation.x = f.angles.j3x;

        // Thumb dual-axis bend + slight inward squeeze
        if (f.isThumb) {
            const ty = -0.28 + t * 0.42;
            f.angles.j1y += (ty - f.angles.j1y) * LF;
            f.j1.rotation.y = f.angles.j1y;
            f.j1.rotation.z = -0.52 + t * 0.12;
        }

        // Natural splay: fingers spread open, converge when curled
        if (!f.isThumb) {
            const sTarget = f.naturalSpread * (1.0 - t * 0.55);
            f.j1.rotation.y += (sTarget - f.j1.rotation.y) * LF;
        }
    });

    // Cache for idle animation reference
    Object.assign(_lastSensors, sensors);
}

/* ------------------------------------------------------------------ */
/*  ANIMATION LOOP  —  zero allocations                                */
/* ------------------------------------------------------------------ */
function _animate() {
    requestAnimationFrame(_animate);

    const dt = _clock ? _clock.getDelta() : 0.016;
    _idleT += dt;
    _scanLineT += dt * 0.9;

    // ── Gentle idle hand sway ──
    if (_handRoot) {
        _handRoot.rotation.y = Math.sin(_idleT * 0.55) * 0.035;
        _handRoot.rotation.z = Math.sin(_idleT * 0.37) * 0.010;
        // Very subtle breathing lift
        _handRoot.position.y = Math.sin(_idleT * 0.42) * 0.06;
    }
    // Subtle wrist motion
    if (_wristBone) {
        _wristBone.rotation.z = Math.sin(_idleT * 0.6) * 0.05;
    }
    // ── Scan ring pulse ──
    const scanRing = _scene.getObjectByName('scanRing');
    if (scanRing) {
        const p = 0.45 + Math.sin(_idleT * 2.8) * 0.15;
        scanRing.material.opacity = p;
        const s = 1.0 + Math.sin(_idleT * 2.2) * 0.04;
        scanRing.scale.set(s, s, s);
    }

    // ── Scan line sweep ──
    const scanLine = _scene.getObjectByName('scanLine');
    if (scanLine) {
        const z = (((_scanLineT % 2.0) / 2.0) * 14) - 7;
        scanLine.position.z = z;
        const op = 0.2 + Math.sin(_scanLineT * Math.PI) * 0.25;
        scanLine.material.opacity = Math.max(0, op);
    }

    // ── Shadow breathes ──
    if (_shadowMesh) {
        _shadowMesh.material.opacity = 0.18 + Math.sin(_idleT * 0.55) * 0.03;
    }

    // ── Glow pulse ──
    if (_glowMesh) {
        const gp = 0.08 + Math.sin(_idleT * 1.1) * 0.04;
        _glowMesh.material.opacity = gp;
    }

    _renderer.render(_scene, _camera);
}

/* ------------------------------------------------------------------ */
/*  RESIZE                                                              */
/* ------------------------------------------------------------------ */
function _onResize() {
    const canvas = _renderer.domElement;
    const parent = canvas.parentElement;
    if (!parent) return;
    const W = parent.clientWidth;
    const H = parent.clientHeight || 380;
    _camera.aspect = W / H;
    _camera.updateProjectionMatrix();
    _renderer.setSize(W, H);
}

/* ------------------------------------------------------------------ */
/*  EXPORTS                                                             */
/* ------------------------------------------------------------------ */
window.updateHand = updateHand;
window.initHand = initHand;

// Auto-init safely after DOM + Three.js are both ready
function _tryInit() {
    if (typeof THREE === 'undefined') {
        setTimeout(_tryInit, 80);
        return;
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initHand);
    } else {
        initHand();
    }
}
_tryInit();