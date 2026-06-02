import os
import pickle
import argparse
import cv2
import numpy as np
from scipy.signal import stft
from joblib import load

MODEL_FILE = "csi_presence_model.joblib"

WINDOW_SECONDS = 1.5
OVERLAP = 0.5
FORCED_FS = 100.0
MAX_FREQ = 10.0

SMOOTH_K = 3
GAUSSIAN_SIGMA = 120


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["valid_csi"], d["valid_csi_timestmp_array"]


def gaussian_blob(h, w, sigma):
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (g / g.max()).astype(np.float32)


def extract_features_and_motion(csi, ts):
    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    _, n = csi.shape
    fs = FORCED_FS

    win = int(WINDOW_SECONDS * fs)
    hop = int(win * (1 - OVERLAP))

    feats = []
    motion = []

    for start in range(0, n - win + 1, hop):
        end = start + win
        seg = csi[:, start:end]

        amp = np.abs(seg)
        sig = amp.mean(axis=0)

        f, _, z = stft(sig, fs=fs, nperseg=min(64, len(sig)))
        p = np.abs(z) ** 2

        mask = f <= MAX_FREQ
        if not np.any(mask):
            continue

        p = p[mask]
        fsel = f[mask]
        total = p.sum() + 1e-12

        low = p[fsel <= 0.5].sum() / total
        mid = p[(fsel > 0.5) & (fsel <= 2.5)].sum() / total
        high = p[fsel > 2.5].sum() / total

        pvec = p.sum(axis=1)
        pvec = pvec / (pvec.sum() + 1e-12)
        ent = -np.sum(pvec * np.log2(pvec + 1e-12))

        feats.append([low, mid, high, ent])
        motion.append(mid + high)

    return np.array(feats), np.array(motion)


def compute_intensity(csi, ts):
    bundle = load(MODEL_FILE)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    feats, motion = extract_features_and_motion(csi, ts)

    if feats.shape[0] == 0:
        raise RuntimeError("No features extracted.")

    Xs = scaler.transform(feats)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(Xs)
        p_occ = probs[:, 1]
    else:
        preds = clf.predict(Xs)
        p_occ = (preds == 1).astype(float)
    print("p_occ min/mean/max:", p_occ.min(), p_occ.mean(), p_occ.max())
    print("motion min/mean/max:", motion.min(), motion.mean(), motion.max())
    # Normalize model probability RELATIVELY within this file
    p_lo = np.percentile(p_occ, 5)
    p_hi = np.percentile(p_occ, 95)
    p_rel = (p_occ - p_lo) / (p_hi - p_lo + 1e-9)
    p_rel = np.clip(p_rel, 0, 1)

    # Normalize motion RELATIVELY within this file
    m_lo = np.percentile(motion, 5)
    m_hi = np.percentile(motion, 95)
    motion_rel = (motion - m_lo) / (m_hi - m_lo + 1e-9)
    motion_rel = np.clip(motion_rel, 0, 1)

    # Blend instead of multiply so it cannot just stay locked at 1
    intensity = 0.35 * p_rel + 0.65 * motion_rel

    # Smooth
    if len(intensity) >= SMOOTH_K:
        intensity = np.convolve(intensity, np.ones(SMOOTH_K) / SMOOTH_K, mode="same")

    return np.clip(intensity, 0, 1)


def overlay_video(pkl_path, video_path, out_path, alpha=0.5):
    csi, ts = load_pkl(pkl_path)
    intensity = compute_intensity(csi, ts)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    blob = gaussian_blob(h, w, GAUSSIAN_SIGMA)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        idx = int(i / total_frames * (len(intensity) - 1))
        inten = intensity[idx]

        heat = np.zeros_like(frame)
        heat[:, :, 2] = (blob * inten * 255).astype(np.uint8)

        out = cv2.addWeighted(frame, 1.0, heat, alpha, 0)

        cv2.putText(out, f"Intensity: {inten:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        writer.write(out)

    cap.release()
    writer.release()

    print("Saved:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_path", required=True)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--out", default="output.mp4")
    args = ap.parse_args()

    overlay_video(args.pkl_path, args.video_path, args.out)


if __name__ == "__main__":
    main()