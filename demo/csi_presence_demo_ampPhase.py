#!/usr/bin/env python3
"""
CSI PRESENCE DEMO (FINAL + PHASE MOTION)

- Amplitude + Doppler -> intensity
- Phase DIFFERENCE -> blob motion
- Single moving blob on black background
"""

import os
import time
import pickle
import numpy as np
import cv2
from scipy.signal import stft
from joblib import load

# ---------------- CONFIG ----------------
MODEL_FILE = "csi_model.joblib"

WINDOW_SECONDS = 1.0
OVERLAP = 0.5
FORCED_FS = 100.0
MAX_FREQ = 8.0

FPS = 6
FRAME_SIZE = 256

# gating + smoothing
P_STABLE_WEIGHT = 0.4
P_VACANT_WEIGHT = 0.6
PRESENCE_ENERGY_THRESH = 0.12
SMOOTH_K = 3
MIN_PERSIST = 3
DISPLAY_THRESH = 0.18

GAUSSIAN_SIGMA = 28

# phase motion
PHASE_GAIN = 6.0          # how much blob moves
PHASE_SMOOTH = 5          # temporal smoothing

LABEL_NAMES = ["VACANT", "STABLE", "ACTIVE"]

# ---------------- LOAD ----------------
def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["valid_csi"], d["valid_csi_timestmp_array"]

# ---------------- FEATURES + PHASE MOTION ----------------
def extract_features_motion_phase(csi, ts):
    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    M, N = csi.shape

    if len(ts) >= 2:
        dt = np.median(np.diff(ts))
        if dt > 1.0:
            dt /= 1000.0
        fs = 1.0 / max(dt, 1e-6)
        if fs < 5 or fs > 5000:
            fs = FORCED_FS
    else:
        fs = FORCED_FS

    win = max(16, int(WINDOW_SECONDS * fs))
    hop = max(1, int(win * (1 - OVERLAP)))

    feats, motion, phase_delta = [], [], []

    if N < win:
        win = N
        hop = N

    for start in range(0, N - win + 1, hop):
        seg = csi[:, start:start + win]

        # -------- amplitude features --------
        amp = np.abs(seg)
        sig = amp.mean(axis=0)

        f, _, Z = stft(sig, fs=fs, nperseg=min(64, len(sig)))
        P = np.abs(Z) ** 2

        mask = f <= MAX_FREQ
        if not np.any(mask):
            continue

        P = P[mask]
        f = f[mask]
        total = P.sum() + 1e-12

        low = P[f <= 0.8].sum() / total
        mid = P[(f > 0.8) & (f <= 2.0)].sum() / total
        high = P[f > 2.0].sum() / total
        ent = -np.sum((P / total) * np.log2(P / total + 1e-12))

        feats.append([low, mid, high, ent])
        motion.append(mid + high)

        # -------- PHASE DIFFERENCE (KEY ADDITION) --------
        phase = np.unwrap(np.angle(seg), axis=1)
        dphi = np.diff(phase, axis=1)
        phase_delta.append(np.mean(np.abs(dphi)))

    return np.array(feats), np.array(motion), np.array(phase_delta)

# ---------------- GAUSSIAN ----------------
def gaussian_blob(size, sigma):
    cx = cy = size // 2
    y, x = np.ogrid[:size, :size]
    g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return g / g.max()

# ---------------- PERSISTENCE ----------------
def persistence(mask, k):
    kernel = np.ones(k)
    return np.convolve(mask.astype(int), kernel, "same") >= k

# ---------------- MAIN ----------------
def main(pkl_path):
    t0 = time.time()

    bundle = load(MODEL_FILE)
    scaler = bundle["scaler"]
    model = bundle["model"]

    csi, ts = load_pkl(pkl_path)
    X, motion, phase_d = extract_features_motion_phase(csi, ts)
    Xs = scaler.transform(X)

    probs = model.predict_proba(Xs)
    p_v, p_s, p_a = probs.T

    # -------- presence intensity --------
    presence = p_a + P_STABLE_WEIGHT * p_s - P_VACANT_WEIGHT * p_v
    presence = np.clip(presence, 0, 1)

    motion_norm = motion / (np.percentile(motion, 90) + 1e-9)
    motion_norm = np.clip(motion_norm, 0, 1)

    I = presence * motion_norm
    I[motion < PRESENCE_ENERGY_THRESH] = 0
    I = np.convolve(I, np.ones(SMOOTH_K)/SMOOTH_K, "same")
    keep = persistence(I >= DISPLAY_THRESH, MIN_PERSIST)
    I[~keep] = 0

    # -------- PHASE-BASED MOTION VECTOR --------
    phase_d = np.convolve(phase_d, np.ones(PHASE_SMOOTH)/PHASE_SMOOTH, "same")
    phase_d = phase_d / (np.percentile(phase_d, 90) + 1e-9)

    angles = np.cumsum(phase_d * PHASE_GAIN)
    dx = np.cos(angles)
    dy = np.sin(angles)

    # -------- video naming --------
    base = os.path.basename(pkl_path)
    name = base.replace(".pkl", "")
    cls = os.path.basename(os.path.dirname(pkl_path))
    out = f"presence_demo_{cls}{name}_phase.mp4"

    writer = cv2.VideoWriter(
        out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_SIZE, FRAME_SIZE),
        True
    )

    blob = gaussian_blob(FRAME_SIZE, GAUSSIAN_SIGMA)
    cx = cy = FRAME_SIZE // 2

    for i, inten in enumerate(I):
        frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)

        if inten > 0:
            cx = int(np.clip(cx + dx[i], 40, FRAME_SIZE-40))
            cy = int(np.clip(cy + dy[i], 40, FRAME_SIZE-40))

            shifted = np.roll(np.roll(blob, cy-FRAME_SIZE//2, axis=0),
                               cx-FRAME_SIZE//2, axis=1)

            frame[:, :, 2] = (shifted * inten * 255).astype(np.uint8)

        label = "VACANT" if inten == 0 else LABEL_NAMES[np.argmax(probs[i])]

        elapsed = time.time() - t0

        cv2.putText(frame, f"File: {base}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        cv2.putText(frame, f"Class: {label}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"I = {inten:.2f}", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame, f"Runtime: {elapsed:.1f}s",
                    (8, FRAME_SIZE-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        writer.write(frame)

    writer.release()
    print(f"[✓] Saved {out}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    args = p.parse_args()
    main(args.pkl)
