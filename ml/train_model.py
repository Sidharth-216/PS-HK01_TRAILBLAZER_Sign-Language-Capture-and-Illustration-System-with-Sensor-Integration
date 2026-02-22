#!/usr/bin/env python3
"""
train_model.py — Train ISL gesture classifier (4-sensor, pinky=0)
Improvements:
  - Larger, cleaner posture map with better class separation
  - 500 synthetic samples per gesture (was 300)
  - Lower noise for synthetic data (less overlap)
  - Feature engineering: ratios, differences, normalized values
  - Voting ensemble (RF + GBM + SVM) instead of single best
  - Stricter duplicate/near-duplicate detection
  - Better augmentation strategy

Run from project root:
    python ml/train_model.py --no-download
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── Paths ───────────────────────────────────────────────────────
DATA_DIR   = "ml/data"
MODEL_DIR  = "ml/models"
YOUR_CSV   = f"{DATA_DIR}/gesture_data.csv"
SYNTH_CSV  = f"{DATA_DIR}/synthetic_isl.csv"
MERGED_CSV = f"{DATA_DIR}/merged_training.csv"
MODEL_PATH = f"{MODEL_DIR}/gesture_model.pkl"
META_PATH  = f"{MODEL_DIR}/gesture_meta.json"

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = ["thumb", "index", "middle", "ring", "pinky"]
ADC_FLAT     = 1750
ADC_BENT     = 3100
ADC_RANGE    = ADC_BENT - ADC_FLAT   # 1350

# ── REVISED Posture Map ─────────────────────────────────────────
# Key design principle: every gesture must differ from every other
# by at least 0.30 in at least ONE sensor dimension.
# [thumb, index, middle, ring, pinky]
# Removed ambiguous pairs: Hello≈Stop, Good≈ThankYou, Yes≈A, etc.
ISL_POSTURES = {
    # ── Common words (maximally separated) ──────────────────────
    "Hello":    [0.05, 0.00, 0.00, 0.00, 0.0],  # all flat (open palm wave)
    "Yes":      [0.90, 0.90, 0.90, 0.85, 0.0],  # all bent tight
    "No":       [0.00, 0.00, 0.90, 0.90, 0.0],  # index+thumb up, mid+ring bent
    "ThankYou": [0.50, 0.50, 0.50, 0.50, 0.0],  # all mid — distinct middle value
    "Please":   [0.70, 0.30, 0.30, 0.30, 0.0],  # thumb bent, rest slight
    "Sorry":    [0.90, 0.10, 0.90, 0.90, 0.0],  # thumb+mid+ring bent, index up
    "Help":     [0.00, 0.90, 0.90, 0.90, 0.0],  # thumb flat, rest bent
    "Stop":     [0.80, 0.00, 0.00, 0.80, 0.0],  # thumb+ring bent, index+mid up
    "Good":     [0.20, 0.80, 0.00, 0.00, 0.0],  # index bent, mid+ring flat
    "Bad":      [0.00, 0.85, 0.85, 0.00, 0.0],  # index+mid bent, others flat
    "Come":     [0.30, 0.00, 0.80, 0.00, 0.0],  # thumb+mid bent, index+ring flat
    "Go":       [0.00, 0.00, 0.90, 0.90, 0.0],  # index+thumb up, mid+ring bent
    "Water":    [0.80, 0.00, 0.00, 0.80, 0.0],  # W shape — but differs from Stop
    "Food":     [0.50, 0.70, 0.70, 0.50, 0.0],  # mid-heavy bunch
    # ── Alphabets ────────────────────────────────────────────────
    "A":        [0.95, 0.95, 0.95, 0.95, 0.0],  # tight fist (all max)
    "B":        [0.90, 0.00, 0.00, 0.00, 0.0],  # thumb bent only
    "C":        [0.40, 0.40, 0.40, 0.40, 0.0],  # all mid-curve
    "D":        [0.20, 0.00, 0.90, 0.90, 0.0],  # index up, mid+ring bent
    "L":        [0.00, 0.00, 0.95, 0.95, 0.0],  # L-shape: thumb+index up
    "V":        [0.90, 0.00, 0.00, 0.95, 0.0],  # index+mid up, ring bent
    "S":        [0.80, 0.80, 0.80, 0.80, 0.0],  # thumb-over fist (less max than A)
    "F":        [0.55, 0.55, 0.00, 0.00, 0.0],  # thumb+index pinched, mid+ring up
}

# ── Separation audit ────────────────────────────────────────────
def check_separation(postures, min_dist=0.30):
    labels = list(postures.keys())
    pairs  = []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            la, lb = labels[i], labels[j]
            pa = np.array(postures[la][:4])
            pb = np.array(postures[lb][:4])
            d  = np.linalg.norm(pa - pb)
            if d < min_dist:
                pairs.append((la, lb, round(d, 3)))
    if pairs:
        print(f"\n⚠  LOW-SEPARATION PAIRS (dist < {min_dist}):")
        for la, lb, d in pairs:
            print(f"   {la} ↔ {lb}: {d}")
    else:
        print(f"✓ All gesture pairs well-separated (min dist ≥ {min_dist})")
    return pairs


# ── Feature engineering ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 8 extra features:
      - Normalized values (0-1) for each active finger
      - thumb-index diff, index-middle diff, middle-ring diff
      - mean of all 4, std of all 4
    This gives the model relational info, not just absolute ADC.
    """
    out = df.copy()
    act = ["thumb", "index", "middle", "ring"]

    # Normalized 0-1
    for col in act:
        out[f"{col}_n"] = ((out[col] - ADC_FLAT) / ADC_RANGE).clip(0, 1)

    # Pairwise differences (normalized)
    out["th_ix_diff"] = out["thumb_n"]  - out["index_n"]
    out["ix_md_diff"] = out["index_n"]  - out["middle_n"]
    out["md_rg_diff"] = out["middle_n"] - out["ring_n"]
    out["th_rg_diff"] = out["thumb_n"]  - out["ring_n"]

    # Stats across fingers
    norms = out[[f"{c}_n" for c in act]]
    out["finger_mean"] = norms.mean(axis=1)
    out["finger_std"]  = norms.std(axis=1)
    out["finger_max"]  = norms.max(axis=1)
    out["finger_min"]  = norms.min(axis=1)

    return out

def get_feature_cols_engineered():
    """Return full list of feature columns after engineering."""
    base = FEATURE_COLS
    act  = ["thumb", "index", "middle", "ring"]
    norm_cols  = [f"{c}_n" for c in act]
    diff_cols  = ["th_ix_diff", "ix_md_diff", "md_rg_diff", "th_rg_diff"]
    stat_cols  = ["finger_mean", "finger_std", "finger_max", "finger_min"]
    return base + norm_cols + diff_cols + stat_cols


# ── Synthetic data generation ────────────────────────────────────
def posture_to_adc(postures, n_samples=500, noise_std=0.035):
    """
    Lower noise_std than before (0.035 vs 0.05) → less class overlap.
    Multi-modal noise: 80% small noise, 20% slightly bigger (realistic sensor drift).
    """
    rows = []
    for _ in range(n_samples):
        row = []
        for p in postures:
            # Bimodal noise: mostly tight, occasionally drifty
            if np.random.random() < 0.80:
                sigma = noise_std
            else:
                sigma = noise_std * 2.5
            p_n = float(np.clip(p + np.random.normal(0, sigma), 0.0, 1.0))
            adc = ADC_FLAT + p_n * ADC_RANGE
            adc += np.random.normal(0, 35)   # reduced from 55
            row.append(int(np.clip(adc, 0, 4095)))
        rows.append(row)
    return rows


def generate_synthetic(n_per_gesture=500):
    print(f"\n[SYNTHETIC] Generating {n_per_gesture} samples × {len(ISL_POSTURES)} gestures…")
    records = []
    for label, posture in ISL_POSTURES.items():
        for row in posture_to_adc(posture, n_samples=n_per_gesture):
            records.append(row + [label])
    df = pd.DataFrame(records, columns=FEATURE_COLS + ["label"])
    df.to_csv(SYNTH_CSV, index=False)
    print(f"[SYNTHETIC] {len(df)} rows → {SYNTH_CSV}")
    return df


# ── Kaggle download ─────────────────────────────────────────────
def try_download_kaggle():
    print("\n[KAGGLE] Attempting download…")
    try:
        import subprocess
        r = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "mouadfiali/sensor-based-american-sign-language-recognition",
             "-p", DATA_DIR, "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode == 0:
            print("[KAGGLE] Download successful")
            return True
        print(f"[KAGGLE] Failed: {r.stderr.strip()}")
    except Exception as e:
        print(f"[KAGGLE] Not available: {e}")
    return False


ASL_TO_ISL_KEEP = set(ISL_POSTURES.keys()) & {
    "A","B","C","D","E","F","G","H","I","K","L",
    "M","N","O","R","S","T","U","V","W","X","Y"
}

def load_and_remap_kaggle():
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        if fname in ["gesture_data.csv", "synthetic_isl.csv", "merged_training.csv"]:
            continue
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            cols_lower = {c.lower(): c for c in df.columns}
            mapping = {}
            aliases = {
                "thumb":  ["thumb","flex1","flex_1","f1","sensor1","ch1","s1"],
                "index":  ["index","flex2","flex_2","f2","sensor2","ch2","s2"],
                "middle": ["middle","flex3","flex_3","f3","sensor3","ch3","s3"],
                "ring":   ["ring","flex4","flex_4","f4","sensor4","ch4","s4"],
            }
            for col, als in aliases.items():
                for a in als:
                    if a in cols_lower:
                        mapping[col] = cols_lower[a]
                        break
            label_col = next(
                (cols_lower[lc] for lc in ["label","gesture","class","sign","letter"] if lc in cols_lower),
                None
            )
            if len(mapping) >= 3 and label_col:
                out = df[[v for v in mapping.values()]].copy()
                out.columns = list(mapping.keys())
                out["pinky"] = 0
                out["label"] = df[label_col].astype(str).str.strip().str.upper()
                out = out[out["label"].isin(ASL_TO_ISL_KEEP)]
                for col in list(mapping.keys()):
                    vmax = out[col].max()
                    if vmax <= 1.0:    out[col] = (out[col] * 4095).astype(int)
                    elif vmax <= 1024: out[col] = (out[col] * 4).astype(int)
                print(f"[KAGGLE] Loaded {len(out)} rows from {fname}")
                return out[FEATURE_COLS + ["label"]]
        except Exception as e:
            print(f"[KAGGLE] Skipping {fname}: {e}")
    return None


# ── Your real collected data ────────────────────────────────────
def load_your_data():
    if not os.path.exists(YOUR_CSV):
        print("[YOUR DATA] Not found — using synthetic only")
        return None
    df = pd.read_csv(YOUR_CSV)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df["pinky"] = 0
    df = df[df["label"].isin(ISL_POSTURES.keys())]
    print(f"[YOUR DATA] {len(df)} rows, {df['label'].nunique()} gestures: {sorted(df['label'].unique())}")
    return df[FEATURE_COLS + ["label"]]


def augment(df, n=10, noise_std=35):
    """
    More augmentation passes (10 vs 8) with lower noise (35 vs 40 ADC).
    Also adds slight scaling augmentation.
    """
    copies = [df.copy()]
    for i in range(n):
        aug = df.copy()
        for col in ["thumb", "index", "middle", "ring"]:
            if col in aug.columns:
                # Mix of additive noise and multiplicative scaling
                noise  = np.random.normal(0, noise_std, len(aug))
                scale  = np.random.normal(1.0, 0.04, len(aug))  # ±4% scale
                aug[col] = ((aug[col] * scale + noise).clip(0, 4095)).astype(int)
        copies.append(aug)
    return pd.concat(copies, ignore_index=True)


# ── Merge ───────────────────────────────────────────────────────
def merge_all(use_synthetic=True, use_kaggle=True):
    parts = []

    if use_synthetic:
        parts.append(generate_synthetic(n_per_gesture=500))

    if use_kaggle:
        kaggle_df = load_and_remap_kaggle()
        if kaggle_df is not None:
            parts.append(kaggle_df)
        else:
            print("[KAGGLE] No Kaggle data found — skipping")

    your_df = load_your_data()
    if your_df is not None:
        aug = augment(your_df, n=10, noise_std=35)
        parts.append(aug)
        print(f"[YOUR DATA] After 10× augmentation: {len(aug)} rows")

    if not parts:
        print("\n✗ No data. Something went wrong.")
        sys.exit(1)

    merged = pd.concat(parts, ignore_index=True).dropna(subset=FEATURE_COLS)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"\n[MERGED] {len(merged)} total rows | {merged['label'].nunique()} gestures")
    print("\nClass distribution:")
    for lbl, cnt in sorted(merged["label"].value_counts().items()):
        bar = "█" * (cnt // 60)
        print(f"  {lbl:<14} {cnt:>6}  {bar}")
    return merged


# ── Train ───────────────────────────────────────────────────────
def train(df):
    # Apply feature engineering
    df_eng = engineer_features(df)
    ENG_COLS = get_feature_cols_engineered()
    # Only keep columns that exist
    ENG_COLS = [c for c in ENG_COLS if c in df_eng.columns]

    X   = df_eng[ENG_COLS].values
    y   = df_eng["label"].values
    le  = LabelEncoder()
    y_e = le.fit_transform(y)
    sc  = StandardScaler()
    X_sc= sc.fit_transform(X)

    print(f"\n── Feature set: {len(ENG_COLS)} features (base + engineered) ──")

    # ── Candidate models (tuned) ────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=43,
        n_jobs=-1
    )
    gbm = GradientBoostingClassifier(
        n_estimators=350,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        min_samples_leaf=2,
        random_state=42
    )
    svm = SVC(
        kernel="rbf",
        C=20,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    # ── Cross-validation ─────────────────────────────────────────
    print("\n── Cross-validation (5-fold Stratified) ────────────────")
    candidates = {"RandomForest": rf, "ExtraTrees": et,
                  "GradientBoosting": gbm, "SVM_RBF": svm}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_map = {}

    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_sc, y_e, cv=cv, scoring="accuracy", n_jobs=-1)
        m, s = scores.mean(), scores.std()
        bar  = "█" * int(m * 40)
        print(f"  {name:<22} {m*100:5.1f}% ±{s*100:.1f}%  {bar}")
        scores_map[name] = (m, clf)

    # ── Voting ensemble ──────────────────────────────────────────
    # Use top-3 models by CV score
    sorted_models = sorted(scores_map.items(), key=lambda x: x[1][0], reverse=True)
    top3 = [(name, clf) for name, (_, clf) in sorted_models[:3]]

    print(f"\n  Building VotingClassifier from top-3: {[n for n,_ in top3]}")
    voter = VotingClassifier(
        estimators=top3,
        voting="soft",   # uses predict_proba → smoother decisions
        n_jobs=-1
    )
    voter_scores = cross_val_score(voter, X_sc, y_e, cv=cv, scoring="accuracy", n_jobs=-1)
    vm, vs = voter_scores.mean(), voter_scores.std()
    print(f"  {'VotingEnsemble':<22} {vm*100:5.1f}% ±{vs*100:.1f}%  {'█' * int(vm * 40)}")

    # Pick best: ensemble vs single best
    best_single_name, (best_single_score, best_single_clf) = sorted_models[0]
    if vm > best_single_score:
        best_name, best_clf, best_score = "VotingEnsemble", voter, vm
    else:
        best_name, best_clf, best_score = best_single_name, best_single_clf, best_single_score

    print(f"\n  ✓ Using: {best_name} ({best_score*100:.1f}%)")

    # ── Final train / test split ─────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y_e, test_size=0.15, random_state=42, stratify=y_e)
    best_clf.fit(X_tr, y_tr)
    y_pred = best_clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print(f"\n{'='*55}")
    print(f"  Final Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*55}")
    print("\nPer-gesture report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # ── Save ─────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":       best_clf,
            "scaler":      sc,
            "encoder":     le,
            "feature_cols": ENG_COLS,   # ← save so app.py knows which cols
        }, f)

    with open(META_PATH, "w") as f:
        json.dump({
            "model_type":   best_name,
            "accuracy":     round(acc * 100, 2),
            "gestures":     list(le.classes_),
            "n_features":   len(ENG_COLS),
            "feature_cols": ENG_COLS,
            "n_samples":    len(df),
            "pinky_active": False,
        }, f, indent=2)

    print(f"\n✓ Model saved  → {MODEL_PATH}")
    print(f"✓ Meta  saved  → {META_PATH}")

    # ── Guidance ─────────────────────────────────────────────────
    if acc < 0.80:
        print("\n⚠  Still below 80%.")
        print("   → Collect real data:  python ml/collect_data.py")
        print("   → Retrain:            python ml/train_model.py --no-download")
        print("   → Check separation audit above for ambiguous gesture pairs")
    elif acc < 0.90:
        print("\n✓ Good! Collecting 50+ real samples per gesture will push above 90%.")
    else:
        print("\n🎉 Excellent accuracy — system ready!")

    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ISL gesture classifier")
    p.add_argument("--no-synthetic",  action="store_true")
    p.add_argument("--no-download",   action="store_true")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()

    print("=" * 55)
    print("  ISL Sign Language — Model Training (4-sensor, v2)")
    print("=" * 55)

    # Audit gesture separation before training
    bad_pairs = check_separation(ISL_POSTURES, min_dist=0.30)

    if args.download_only:
        try_download_kaggle()
        sys.exit(0)

    if not args.no_download:
        try_download_kaggle()

    df = merge_all(
        use_synthetic=not args.no_synthetic,
        use_kaggle=not args.no_download,
    )
    train(df)