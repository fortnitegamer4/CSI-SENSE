import os
import pickle
import argparse
import numpy as np
from joblib import load
from scipy.signal import stft

MODEL_FILE = "csi_presence_model.joblib"

WINDOW_SECONDS = 1.5
OVERLAP = 0.5
FORCED_FS = 100.0
MAX_FREQ = 10.0

FILE_FRAC_HIGH = 0.15
FILE_MEAN_PROB = 0.5

INV_LABEL = {0: "VACANT", 1: "OCCUPIED"}


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["valid_csi"], d["valid_csi_timestmp_array"]


def safe_fs_from_ts(ts):
    if len(ts) >= 2:
        dt = np.median(np.diff(ts))
        if dt > 1.0:
            dt /= 1000.0
        fs = 1.0 / max(dt, 1e-12)
        if fs < 5 or fs > 5000:
            fs = FORCED_FS
    else:
        fs = FORCED_FS
    return fs


def extract_windows_features(csi, ts, window_seconds=WINDOW_SECONDS, overlap=OVERLAP, use_phase=False):
    if csi.ndim != 2:
        raise ValueError(f"Unexpected CSI array dims: {csi.shape}")
    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    _, n = csi.shape
    if n < 4:
        return np.zeros((0, 4 + (1 if use_phase else 0))), np.zeros((0,)), []

    fs = safe_fs_from_ts(ts)
    win = max(16, int(round(window_seconds * fs)))
    hop = max(1, int(round(win * (1.0 - overlap))))

    if n < win:
        win = n
        hop = n

    feats = []
    motion = []
    slices = []

    for start in range(0, n - win + 1, hop):
        end = start + win
        seg_complex = csi[:, start:end]
        seg_amp = np.abs(seg_complex)
        sig = seg_amp.mean(axis=0).astype(float)

        if not np.all(np.isfinite(sig)):
            continue
        if np.allclose(sig, sig[0]):
            continue

        try:
            f, _, z = stft(sig, fs=fs, nperseg=min(64, len(sig)))
        except Exception:
            m = len(sig)
            yf = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
            f = np.fft.rfftfreq(m, 1.0 / fs) if m > 1 else np.array([0.0])
            z = np.zeros((len(f), m), dtype=complex)
            z[:, 0:len(yf)] = np.atleast_2d(yf).T

        p = np.abs(z) ** 2
        mask = f <= MAX_FREQ
        if not np.any(mask):
            continue

        p = p[mask]
        fsel = f[mask]

        if not np.all(np.isfinite(p)):
            continue

        total = p.sum() + 1e-12
        if total <= 1e-12 or not np.isfinite(total):
            continue

        low = float(p[fsel <= 0.5].sum() / total)
        mid = float(p[(fsel > 0.5) & (fsel <= 2.5)].sum() / total)
        high = float(p[fsel > 2.5].sum() / total)

        pvec = p.sum(axis=1)
        pvec = pvec / (pvec.sum() + 1e-12)
        ent = float(-np.sum(pvec * np.log2(pvec + 1e-12)))

        feat = [low, mid, high, ent]

        if use_phase:
            try:
                ph = np.angle(seg_complex)
                ph_un = np.unwrap(ph, axis=1)
                dph = np.diff(ph_un, axis=1)
                var_per_sub = np.var(dph, axis=1)
                phase_var = float(np.mean(var_per_sub))
            except Exception:
                phase_var = 0.0
            feat.append(phase_var)

        feats.append(feat)
        motion.append(mid + high)
        slices.append((start, end))

    if not feats:
        return np.zeros((0, 4 + (1 if use_phase else 0))), np.zeros((0,)), []

    return np.array(feats, dtype=float), np.array(motion, dtype=float), slices


def classify_one_file(pkl_path, model_path=MODEL_FILE, use_phase=False):
    bundle = load(model_path)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    csi, ts = load_pkl(pkl_path)
    feats, motion, slices = extract_windows_features(csi, ts, use_phase=use_phase)

    if feats.shape[0] == 0:
        raise RuntimeError("No valid windows extracted from this PKL.")

    xs = scaler.transform(feats)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(xs)
        if probs.shape[1] == 1:
            p_occ = np.zeros(len(xs), dtype=float)
        else:
            p_occ = probs[:, 1]
    else:
        preds = clf.predict(xs)
        p_occ = (preds == 1).astype(float)

    high_conf = p_occ >= 0.6
    frac_high = float(high_conf.mean()) if p_occ.size else 0.0
    mean_prob = float(p_occ.mean()) if p_occ.size else 0.0

    if frac_high >= FILE_FRAC_HIGH or mean_prob >= FILE_MEAN_PROB:
        pred_label = 1
    else:
        pred_label = 0

    print(f"File: {os.path.basename(pkl_path)}")
    print(f"Windows extracted: {len(feats)}")
    print(f"Mean occupied probability: {mean_prob:.4f}")
    print(f"Fraction of high-confidence occupied windows: {frac_high:.4f}")
    print(f"Prediction: {INV_LABEL[pred_label]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl_path", required=True, help="Path to one PKL file")
    ap.add_argument("--model_path", default=MODEL_FILE, help="Path to trained joblib model")
    ap.add_argument("--use_phase", action="store_true", help="Use phase feature if model expects it")
    args = ap.parse_args()

    classify_one_file(args.pkl_path, model_path=args.model_path, use_phase=args.use_phase)


if __name__ == "__main__":
    main()