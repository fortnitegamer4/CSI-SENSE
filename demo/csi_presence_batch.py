#!/usr/bin/env python3
"""
source venv/bin/activate

Training:
  python demo/csi_presence_batch.py \
    --data_root collected_dataset \
    --retrain

  python demo/csi_presence_batch.py \
    --data_root external_data/processed/combined_presence \
    --retrain \
    --mmfi_limit 50 \
    --wipose_limit 50

Generate videos:
  python demo/csi_presence_batch.py \
    --data_root collected_dataset \
    --use_phase
"""

import os
import time
import pickle
import argparse
from glob import glob

import numpy as np
import cv2
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load
from collections import Counter

MODEL_FILE = "csi_presence_model.joblib"

# windowing / spectral params
WINDOW_SECONDS = 1.5
OVERLAP = 0.5
FORCED_FS = 100.0
MAX_FREQ = 10.0

# video rendering params
FPS = 6
FRAME_SIZE = 256
GAUSSIAN_SIGMA = 28

# intensity smoothing / persistence
SMOOTH_K = 3
MIN_PERSIST = 3
DISPLAY_THRESH = 0.12

# decision thresholds
WINDOW_OCC_THRESH = 0.40
FILE_FRAC_HIGH = 0.15
FILE_MEAN_PROB = 0.5

# labels: binary
LABELS_MAP = {"vacant": 0, "stable": 1, "active": 1, "occupied": 1}
INV_LABEL = {0: "VACANT", 1: "OCCUPIED"}


# ---------------- helpers ----------------
def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    csi = d["valid_csi"]
    ts = d["valid_csi_timestmp_array"]
    return csi, ts


def gaussian_blob(size, sigma):
    cx = cy = size // 2
    y, x = np.ogrid[:size, :size]
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (g / g.max()).astype(np.float32)


def persistence(mask, k):
    kernel = np.ones(k)
    return np.convolve(mask.astype(int), kernel, mode="same") >= k


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


# ---------------- feature extraction ----------------
def extract_windows_features(csi, ts, window_seconds=WINDOW_SECONDS, overlap=OVERLAP, use_phase=False):
    if csi.ndim != 2:
        raise ValueError("Unexpected CSI array dims")
    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    M, N = csi.shape
    if N < 4:
        return np.zeros((0, 4 + (1 if use_phase else 0))), np.zeros((0,)), []

    fs = safe_fs_from_ts(ts)
    win = max(16, int(round(window_seconds * fs)))
    hop = max(1, int(round(win * (1.0 - overlap))))

    if N < win:
        win = N
        hop = N

    feats = []
    motion = []
    slices = []
    for start in range(0, N - win + 1, hop):
        end = start + win
        seg_complex = csi[:, start:end]
        seg_amp = np.abs(seg_complex)
        sig = seg_amp.mean(axis=0).astype(float)

        try:
            f, _, Z = stft(sig, fs=fs, nperseg=min(64, len(sig)))
        except Exception:
            n = len(sig)
            yf = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
            f = np.fft.rfftfreq(n, 1.0 / fs) if n > 1 else np.array([0.0])
            Z = np.zeros((len(f), n), dtype=complex)
            Z[:, 0:len(yf)] = np.atleast_2d(yf).T

        P = np.abs(Z) ** 2
        mask = f <= MAX_FREQ
        if not np.any(mask):
            continue
        P = P[mask]
        fsel = f[mask]
        total = P.sum() + 1e-12

        low = float(P[fsel <= 0.5].sum() / total)
        mid = float(P[(fsel > 0.5) & (fsel <= 2.5)].sum() / total)
        high = float(P[fsel > 2.5].sum() / total)

        pvec = P.sum(axis=1)
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

    if len(feats) == 0:
        return np.zeros((0, 4 + (1 if use_phase else 0))), np.zeros((0,)), []
    return np.array(feats, dtype=float), np.array(motion, dtype=float), slices


# ---------------- training pipeline ----------------
def build_dataset_windows(data_root, use_phase=False, verbose=True):
    X = []
    y = []
    root = os.path.abspath(data_root)
    for building in sorted(os.listdir(root)):
        bdir = os.path.join(root, building)
        if not os.path.isdir(bdir):
            continue
        for label_dir in sorted(os.listdir(bdir)):
            ldir = os.path.join(bdir, label_dir)
            if not os.path.isdir(ldir):
                continue
            label = LABELS_MAP.get(label_dir.lower(), 1)
            for pkl in sorted(glob(os.path.join(ldir, "*.pkl"))):
                try:
                    csi, ts = load_pkl(pkl)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] could not load {pkl}: {e}")
                    continue
                feats, motion, slices = extract_windows_features(csi, ts, use_phase=use_phase)
                if feats.shape[0] == 0:
                    continue
                for row in feats:
                    X.append(row)
                    y.append(label)
                if verbose:
                    print(f"[LOAD] {os.path.relpath(pkl, root)} -> windows={feats.shape[0]}")
    if len(X) == 0:
        return np.zeros((0, 4 + (1 if use_phase else 0))), np.array([], dtype=int)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def collect_files(data_root, mmfi_limit=None, wipose_limit=None):
    files = []
    labels = []

    root = os.path.abspath(data_root)

    mmfi_seen = 0
    wipose_seen = 0

    def add_files_from_label_dir(building_name, label_dir_path, label_name):
        nonlocal mmfi_seen, wipose_seen, files, labels

        label = LABELS_MAP.get(label_name.lower(), 1)

        for pkl in sorted(glob(os.path.join(label_dir_path, "*.pkl"))):
            bname = building_name.lower()

            if bname == "mmfi" and mmfi_limit is not None:
                if mmfi_seen >= mmfi_limit:
                    continue
                mmfi_seen += 1

            if bname == "wipose" and wipose_limit is not None:
                if wipose_seen >= wipose_limit:
                    continue
                wipose_seen += 1

            files.append(pkl)
            labels.append(label)

    for building in sorted(os.listdir(root)):
        bdir = os.path.join(root, building)
        if not os.path.isdir(bdir):
            continue

        # Case 1: standard layout
        # data_root/BuildingA/vacant/*.pkl
        direct_label_dirs = []
        nested_building_dirs = []

        for child in sorted(os.listdir(bdir)):
            child_path = os.path.join(bdir, child)
            if not os.path.isdir(child_path):
                continue

            if child.lower() in LABELS_MAP:
                direct_label_dirs.append((child, child_path))
            else:
                nested_building_dirs.append((child, child_path))

        if direct_label_dirs:
            for label_name, label_path in direct_label_dirs:
                add_files_from_label_dir(building, label_path, label_name)
            continue

        # Case 2: nested layout like Original/BuildingA/vacant/*.pkl
        for nested_building, nested_bdir in nested_building_dirs:
            for label_name in sorted(os.listdir(nested_bdir)):
                label_path = os.path.join(nested_bdir, label_name)
                if not os.path.isdir(label_path):
                    continue
                if label_name.lower() not in LABELS_MAP:
                    continue

                # treat nested building name as the real building source
                # unless parent folder is MMFi/WiPose, which already use direct layout
                add_files_from_label_dir(nested_building, label_path, label_name)

    return files, labels

def build_from_file_list(file_list, use_phase=False):
    X, y = [], []

    for pkl, label in file_list:
        try:
            csi, ts = load_pkl(pkl)
        except Exception:
            continue

        feats, _, _ = extract_windows_features(csi, ts, use_phase=use_phase)
        for row in feats:
            X.append(row)
            y.append(label)

    return np.array(X), np.array(y)


def evaluate_file_list(file_list, scaler, clf, use_phase=False, verbose=True):
    totals = {"vacant": 0, "occupied": 0}
    corrects = {"vacant": 0, "occupied": 0}
    perfile_results = []

    for pkl_path, actual_label in file_list:
        actual_name = "vacant" if actual_label == 0 else "occupied"
        totals[actual_name] += 1

        try:
            csi, ts = load_pkl(pkl_path)
        except Exception as e:
            if verbose:
                print(f"[WARN] skipping {pkl_path}: load failed: {e}")
            continue

        feats, motion, slices = extract_windows_features(csi, ts, use_phase=use_phase)
        if feats.shape[0] == 0:
            if verbose:
                print(f"[WARN] {os.path.basename(pkl_path)} -> no windows extracted, skipping.")
            perfile_results.append((pkl_path, actual_label, None))
            continue

        Xs = scaler.transform(feats)
        probs = clf.predict_proba(Xs) if hasattr(clf, "predict_proba") else None

        if probs is not None:
            p_occ = probs[:, 1]
        else:
            preds = clf.predict(Xs)
            p_occ = (preds == 1).astype(float)

        if p_occ.size > 0:
            high_conf = p_occ >= 0.6
            frac_high = high_conf.mean()
            mean_prob = p_occ.mean()
            if frac_high >= FILE_FRAC_HIGH or mean_prob >= FILE_MEAN_PROB:
                pred_label = 1
            else:
                pred_label = 0
        else:
            pred_label = 0

        if pred_label == actual_label:
            corrects[actual_name] += 1

        perfile_results.append((pkl_path, actual_label, pred_label))

        if verbose:
            print(
                f"[TEST FILE] {os.path.basename(pkl_path)} -> "
                f"actual={INV_LABEL[actual_label]} predicted={INV_LABEL[pred_label]}"
            )

    total_files = sum(totals.values())
    correct_total = sum(corrects.values())
    overall_acc = correct_total / total_files if total_files else 0.0

    print("\n===== TEST FILE-LEVEL ACCURACY REPORT =====")
    print(f"Overall: {correct_total}/{total_files} = {overall_acc:.3f}\n")
    for cls_name in ["vacant", "occupied"]:
        tot = totals[cls_name]
        cor = corrects[cls_name]
        acc = cor / tot if tot else 0.0
        print(f"{cls_name.upper():7s}: {cor}/{tot} = {acc:.3f}")

    y_true = []
    y_pred = []
    for _, a, p in perfile_results:
        if p is None:
            continue
        y_true.append(a)
        y_pred.append(p)

    if y_true:
        print("\nClassification report (file-level, test split):")
        print(classification_report(y_true, y_pred, target_names=["vacant", "occupied"]))


def train_and_save_model(data_root, use_phase=False, mmfi_limit=None, wipose_limit=None):
    print("[*] Collecting files...")

    files, labels = collect_files(
        data_root,
        mmfi_limit=mmfi_limit,
        wipose_limit=wipose_limit
    )

    print(f"Total selected files: {len(files)}")
    print(f"MMFi limit: {mmfi_limit}")
    print(f"WiPose limit: {wipose_limit}")

    train_files, test_files, train_labels, test_labels = train_test_split(
        files,
        labels,
        test_size=0.4,
        stratify=labels,
        random_state=42
    )
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    train_items = list(zip(train_files, train_labels))
    test_items = list(zip(test_files, test_labels))

    print("[*] Building TRAIN dataset (windows)...")
    X, y = build_from_file_list(train_items, use_phase=use_phase)

    if X.shape[0] == 0:
        raise RuntimeError("No training windows found. Check data_root and segmentation parameters.")

    print(f"  -> train windows: {len(y)}  features: {X.shape[1]}")

    counts = Counter(y)
    print("Class counts before downsampling:", counts)
    max_per_class = min(counts.values())
    idx_keep = []
    kept = {0: 0, 1: 0}
    for i, label in enumerate(y):
        if kept[label] < max_per_class:
            idx_keep.append(i)
            kept[label] += 1
    X = X[idx_keep]
    y = y[idx_keep]
    print("Class counts after downsampling:", Counter(y))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("[*] Cross-validating (f1_macro)...")
    try:
        scores = cross_val_score(clf, Xs, y, cv=5, scoring="f1_macro", n_jobs=-1)
        print(f"  CV macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    except Exception:
        print("  CV failed or skipped (small dataset).")

    print("[*] Training final model...")
    clf.fit(Xs, y)

    print("\n[*] Evaluating on TEST FILES at file-level...")
    evaluate_file_list(test_items, scaler, clf, use_phase=use_phase, verbose=True)

    dump(
        {
            "scaler": scaler,
            "model": clf,
            "use_phase": use_phase,
            "train_files": train_files,
            "test_files": test_files,
            "mmfi_limit": mmfi_limit,
            "wipose_limit": wipose_limit,
        },
        MODEL_FILE
    )
    print(f"[+] Saved model -> {MODEL_FILE}")


# ---------------- inference / video generation ----------------
def process_all_pkls(data_root, outdir="outputs", use_phase=False, fps=FPS):
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model file not found. Run with --retrain to train first.")
    bundle = load(MODEL_FILE)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    root = os.path.abspath(data_root)
    os.makedirs(outdir, exist_ok=True)

    pkl_list = []
    for building in sorted(os.listdir(root)):
        bdir = os.path.join(root, building)
        if not os.path.isdir(bdir):
            continue
        for label_dir in sorted(os.listdir(bdir)):
            ldir = os.path.join(bdir, label_dir)
            if not os.path.isdir(ldir):
                continue
            for pkl in sorted(glob(os.path.join(ldir, "*.pkl"))):
                pkl_list.append((pkl, label_dir.lower()))

    if not pkl_list:
        print("[!] No pkl files found.")
        return

    totals = {"vacant": 0, "occupied": 0}
    corrects = {"vacant": 0, "occupied": 0}
    perfile_results = []

    blob = gaussian_blob(FRAME_SIZE, GAUSSIAN_SIGMA)

    for pkl_path, actual_dir in pkl_list:
        actual_label = 0 if actual_dir == "vacant" else 1
        totals["vacant" if actual_label == 0 else "occupied"] += 1

        base = os.path.basename(pkl_path)
        name = base.replace(".pkl", "")
        suffix = "_phase" if use_phase else ""
        out_subdir = os.path.join(outdir, "vacant" if actual_label == 0 else "occupied")
        os.makedirs(out_subdir, exist_ok=True)
        outname = f"presence_demo_{('vacant' if actual_label == 0 else 'occupied')}{name}{suffix}.mp4"
        outpath = os.path.join(out_subdir, outname)

        try:
            csi, ts = load_pkl(pkl_path)
        except Exception as e:
            print(f"[WARN] skipping {pkl_path}: load failed: {e}")
            continue

        feats, motion, slices = extract_windows_features(csi, ts, use_phase=use_phase)
        if feats.shape[0] == 0:
            print(f"[WARN] {base} -> no windows extracted, skipping video.")
            perfile_results.append((pkl_path, actual_label, None))
            continue

        Xs = scaler.transform(feats)
        probs = clf.predict_proba(Xs) if hasattr(clf, "predict_proba") else None

        if probs is not None:
            p_occ = probs[:, 1]
            preds = (p_occ >= WINDOW_OCC_THRESH).astype(int)
        else:
            preds = clf.predict(Xs)
            p_occ = (preds == 1).astype(float)

        if p_occ.size > 0:
            high_conf = p_occ >= 0.6
            frac_high = high_conf.mean()
            mean_prob = p_occ.mean()
            if frac_high >= FILE_FRAC_HIGH or mean_prob >= FILE_MEAN_PROB:
                pred_label = 1
            else:
                pred_label = 0
        else:
            pred_label = 0

        if pred_label == actual_label:
            corrects["vacant" if actual_label == 0 else "occupied"] += 1

        perfile_results.append((pkl_path, actual_label, pred_label))

        motion_p90 = np.percentile(motion, 90) if motion.size else 1.0
        motion_norm = np.clip(motion / (motion_p90 + 1e-9), 0.0, 1.0)
        motion_floor = 0.25
        intensity = p_occ * (motion_floor + 0.75 * motion_norm)

        intensity = np.convolve(intensity, np.ones(SMOOTH_K) / SMOOTH_K, mode="same")
        keep = persistence(intensity >= DISPLAY_THRESH, MIN_PERSIST)
        intensity[~keep] = 0.0
        intensity = np.clip(intensity, 0.0, 1.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(outpath, fourcc, fps, (FRAME_SIZE, FRAME_SIZE), True)

        t0 = time.time()
        for i, inten in enumerate(intensity):
            frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            if inten > 0:
                frame[:, :, 2] = (blob * inten * 255.0).astype(np.uint8)
            label_str = INV_LABEL[pred_label]
            pstr = f"{float(p_occ[i]):.2f}" if p_occ.size > i else f"{p_occ.mean():.2f}"
            cv2.putText(frame, f"File: {base}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(frame, f"Actual: {INV_LABEL[actual_label]}", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pred: {label_str} ({pstr})", (8, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elapsed = time.time() - t0
            cv2.putText(frame, f"Runtime: {elapsed:.1f}s", (8, FRAME_SIZE - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            writer.write(frame)
        writer.release()

        print(
            f"[DONE] {os.path.relpath(pkl_path, root)} -> "
            f"actual={INV_LABEL[actual_label]} predicted={INV_LABEL[pred_label]} saved={outpath}"
        )

    total_files = sum(totals.values())
    correct_total = sum(corrects.values())
    overall_acc = correct_total / total_files if total_files else 0.0
    print("\n===== ACCURACY REPORT =====")
    print(f"Overall: {correct_total}/{total_files} = {overall_acc:.3f}\n")
    for cls_name in ["vacant", "occupied"]:
        tot = totals[cls_name]
        cor = corrects[cls_name]
        acc = cor / tot if tot else 0.0
        print(f"{cls_name.upper():7s}: {cor}/{tot} = {acc:.3f}")

    y_true = []
    y_pred = []
    for _, a, p in perfile_results:
        if p is None:
            continue
        y_true.append(a)
        y_pred.append(p)
    if y_true:
        print("\nClassification report (file-level):")
        print(classification_report(y_true, y_pred, target_names=["vacant", "occupied"]))


# ----------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="root folder containing building/dataset subfolders")
    p.add_argument("--retrain", action="store_true",
                   help="retrain model from .pkl windows and save")
    p.add_argument("--use_phase", action="store_true",
                   help="include phase-derived feature")
    p.add_argument("--outdir", default="outputs",
                   help="output folder for mp4s")
    p.add_argument("--fps", type=int, default=FPS,
                   help="fps for generated mp4s")
    p.add_argument("--mmfi_limit", type=int, default=None,
                   help="max number of MMFi occupied files to use")
    p.add_argument("--wipose_limit", type=int, default=None,
                   help="max number of WiPose occupied files to use")
    args = p.parse_args()

    if args.retrain:
        train_and_save_model(
            args.data_root,
            use_phase=args.use_phase,
            mmfi_limit=args.mmfi_limit,
            wipose_limit=args.wipose_limit
        )
        return

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found; run with --retrain to train first.")

    process_all_pkls(args.data_root, outdir=args.outdir,
                     use_phase=args.use_phase, fps=args.fps)


if __name__ == "__main__":
    main()