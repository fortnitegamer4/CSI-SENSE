import os
import csv
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from joblib import load

MODEL_FILE = "csi_presence_model.joblib"

WINDOW_SECONDS = 1.5
OVERLAP = 0.5
FORCED_FS = 100.0
MAX_FREQ = 10.0

FILE_FRAC_HIGH = 0.15
FILE_MEAN_PROB = 0.5

LABELS_MAP = {"vacant": 0, "stable": 1, "active": 1, "occupied": 1}
INV_LABEL = {0: "VACANT", 1: "OCCUPIED"}


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["valid_csi"], d["valid_csi_timestmp_array"]


def true_label_from_path(path):
    parts = [p.lower() for p in path.split(os.sep)]
    for p in reversed(parts):
        if p in LABELS_MAP:
            return LABELS_MAP[p]
    return None


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


def extract_windows_features(csi, ts):
    if csi.ndim != 2:
        raise ValueError(f"Unexpected CSI shape: {csi.shape}")

    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    _, n = csi.shape
    if n < 4:
        return np.zeros((0, 4)), np.zeros((0,))

    fs = safe_fs_from_ts(ts)
    win = max(16, int(round(WINDOW_SECONDS * fs)))
    hop = max(1, int(round(win * (1.0 - OVERLAP))))

    if n < win:
        win = n
        hop = n

    feats = []
    motion = []

    for start in range(0, n - win + 1, hop):
        end = start + win
        seg = csi[:, start:end]
        amp = np.abs(seg)
        sig = amp.mean(axis=0).astype(float)

        if not np.all(np.isfinite(sig)):
            continue

        f, _, z = stft(sig, fs=fs, nperseg=min(64, len(sig)))
        p = np.abs(z) ** 2

        mask = f <= MAX_FREQ
        if not np.any(mask):
            continue

        p = p[mask]
        fsel = f[mask]
        total = p.sum() + 1e-12

        low = float(p[fsel <= 0.5].sum() / total)
        mid = float(p[(fsel > 0.5) & (fsel <= 2.5)].sum() / total)
        high = float(p[fsel > 2.5].sum() / total)

        pvec = p.sum(axis=1)
        pvec = pvec / (pvec.sum() + 1e-12)
        ent = float(-np.sum(pvec * np.log2(pvec + 1e-12)))

        feats.append([low, mid, high, ent])
        motion.append(mid + high)

    return np.array(feats, dtype=float), np.array(motion, dtype=float)


def classify_file(path, scaler, clf):
    csi, ts = load_pkl(path)
    feats, _ = extract_windows_features(csi, ts)

    if feats.shape[0] == 0:
        return None, None, None, 0

    xs = scaler.transform(feats)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(xs)
        p_occ = probs[:, 1]
    else:
        preds = clf.predict(xs)
        p_occ = (preds == 1).astype(float)

    frac_high = float((p_occ >= 0.6).mean())
    mean_prob = float(p_occ.mean())

    pred = 1 if (frac_high >= FILE_FRAC_HIGH or mean_prob >= FILE_MEAN_PROB) else 0

    return pred, mean_prob, frac_high, len(feats)


def spectral_entropy(sig):
    f, _, z = stft(sig, fs=FORCED_FS, nperseg=min(64, len(sig)))
    p = np.abs(z) ** 2
    mask = f <= MAX_FREQ
    if not np.any(mask):
        return 0.0

    p = p[mask]
    pvec = p.sum(axis=1)
    pvec = pvec / (pvec.sum() + 1e-12)
    return float(-np.sum(pvec * np.log2(pvec + 1e-12)))


def complexity_features(path):
    csi, ts = load_pkl(path)

    if csi.ndim != 2:
        raise ValueError(f"Bad CSI shape: {csi.shape}")

    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    amp = np.abs(csi).astype(float)
    amp = amp[np.all(np.isfinite(amp), axis=1)]

    if amp.size == 0:
        raise ValueError("No finite CSI values")

    mean_amp = float(np.mean(amp))
    amp_std = float(np.std(amp))

    temporal_diff = np.abs(np.diff(amp, axis=1))
    mean_temporal_change = float(np.mean(temporal_diff))

    subcarrier_spread = float(np.mean(np.std(amp, axis=0)))
    time_instability = float(np.mean(np.std(amp, axis=1)))

    sig = amp.mean(axis=0)
    ent = spectral_entropy(sig)

    # Higher score = more distorted/unstable signal.
    # Do NOT include "occupied probability" here, or it becomes circular.
    return [
        amp_std / (mean_amp + 1e-9),
        mean_temporal_change / (mean_amp + 1e-9),
        subcarrier_spread / (mean_amp + 1e-9),
        time_instability / (mean_amp + 1e-9),
        ent,
    ]


def assign_complexity_occupied_only(occupied_files):
    rows = []
    valid = []

    for f in occupied_files:
        try:
            rows.append(complexity_features(f))
            valid.append(f)
        except Exception as e:
            print(f"[SKIP complexity] {f}: {e}")

    vals = np.array(rows, dtype=float)
    vals_scaled = StandardScaler().fit_transform(vals)

    score = (
        0.25 * vals_scaled[:, 0] +
        0.25 * vals_scaled[:, 1] +
        0.20 * vals_scaled[:, 2] +
        0.15 * vals_scaled[:, 3] +
        0.15 * vals_scaled[:, 4]
    )

    # Force exactly-ish thirds among occupied files only
    order = np.argsort(score)
    n = len(order)

    comp = {}
    score_map = {}

    for rank, idx in enumerate(order):
        f = os.path.abspath(valid[idx])
        s = float(score[idx])

        if rank < n / 3:
            lab = "simple"
        elif rank < 2 * n / 3:
            lab = "moderate"
        else:
            lab = "complex"

        comp[f] = lab
        score_map[f] = s

    return comp, score_map


def make_graph(summary, out_path):
    labels = ["baseline", "simple", "moderate", "complex"]
    display = ["Baseline\nvacant", "Simple\noccupied", "Moderate\noccupied", "Complex\noccupied"]

    accs = [summary[x]["accuracy"] for x in labels]
    counts = [summary[x]["total"] for x in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x, accs)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("File-Level Accuracy")
    ax.set_title("CSI-Sense Accuracy: Baseline vs Estimated Obstruction Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{display[i]}\n(n={counts[i]})" for i in range(len(labels))])
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, v in enumerate(accs):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    ax.text(
        0.5, -0.23,
        "Baseline uses vacant files. Complexity groups use occupied files only and are estimated from CSI signal distortion.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[+] Saved graph: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model_path", default=MODEL_FILE)
    ap.add_argument("--all_files", action="store_true")
    ap.add_argument("--csv_out", default="baseline_complexity_results.csv")
    ap.add_argument("--graph_out", default="baseline_complexity_graph.png")
    args = ap.parse_args()

    bundle = load(args.model_path)
    scaler = bundle["scaler"]
    clf = bundle["model"]

    all_files = sorted(glob(os.path.join(args.root, "**", "*.pkl"), recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    if not all_files:
        print("No .pkl files found.")
        return

    if args.all_files:
        eval_files = all_files
        print("[*] Evaluating ALL files. This may include training files.")
    else:
        if "test_files" not in bundle:
            print("[!] Model has no saved test_files. Use --all_files.")
            return

        test_set = set(os.path.abspath(f) for f in bundle["test_files"])
        eval_files = [f for f in all_files if f in test_set]

        if not eval_files:
            print("[!] No saved test files matched current paths. Use --all_files.")
            return

        print(f"[*] Evaluating saved held-out test files only: {len(eval_files)}")

    labeled = []
    for f in eval_files:
        actual = true_label_from_path(f)
        if actual is None:
            print(f"[SKIP label] Cannot infer label: {f}")
            continue
        labeled.append((f, actual))

    vacant_files = [f for f, y in labeled if y == 0]
    occupied_files = [f for f, y in labeled if y == 1]

    complexity_map, score_map = assign_complexity_occupied_only(occupied_files)

    results = []

    for f, actual in labeled:
        pred, mean_prob, frac_high, windows = classify_file(f, scaler, clf)
        if pred is None:
            print(f"[SKIP classify] No windows: {f}")
            continue

        if actual == 0:
            group = "baseline"
            comp_score = ""
        else:
            group = complexity_map.get(os.path.abspath(f), "unknown")
            comp_score = score_map.get(os.path.abspath(f), "")

        results.append({
            "file": f,
            "group": group,
            "complexity_score": comp_score,
            "actual": actual,
            "pred": pred,
            "correct": int(actual == pred),
            "mean_p_occ": mean_prob,
            "frac_high": frac_high,
            "windows": windows,
        })

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "group", "complexity_score",
            "actual", "pred", "correct", "mean_p_occ", "frac_high", "windows"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"[+] Saved CSV: {args.csv_out}")

    summary = {}
    for group in ["baseline", "simple", "moderate", "complex"]:
        g = [r for r in results if r["group"] == group]
        total = len(g)
        correct = sum(r["correct"] for r in g)
        acc = correct / total if total else 0.0
        summary[group] = {"total": total, "correct": correct, "accuracy": acc}

    print("\n===== ACCURACY SUMMARY =====")
    for group in ["baseline", "simple", "moderate", "complex"]:
        s = summary[group]
        print(f"{group.upper():9s}: {s['correct']}/{s['total']} = {s['accuracy']:.3f}")

    print("\n===== GROUP MEAN OCCUPIED PROBABILITY =====")
    for group in ["baseline", "simple", "moderate", "complex"]:
        g = [r for r in results if r["group"] == group]
        if not g:
            print(f"{group.upper():9s}: NA")
        else:
            print(f"{group.upper():9s}: {np.mean([r['mean_p_occ'] for r in g]):.3f}")

    make_graph(summary, args.graph_out)


if __name__ == "__main__":
    main()