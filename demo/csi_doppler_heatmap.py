#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# ---------------- CONFIG ----------------
MAX_FREQ = 8.0          # Hz
NFFT = 128
WINDOW_SECONDS = 1.0
OVERLAP = 0.75

# ---------------- LOAD ----------------
def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["valid_csi"], d["valid_csi_timestmp_array"]

# ---------------- HEATMAP ----------------
def plot_doppler_heatmap(pkl_path):
    csi, ts = load_pkl(pkl_path)

    print("CSI shape:", csi.shape)

    # force (subcarriers, samples)
    if csi.shape[0] > csi.shape[1]:
        csi = csi.T

    # sampling rate
    dt = np.median(np.diff(ts))
    if dt > 1:
        dt /= 1000.0
    fs = 1.0 / max(dt, 1e-6)

    print(f"fs = {fs:.2f} Hz")

    # magnitude CSI averaged across subcarriers
    signal = np.abs(csi).mean(axis=0)
    signal -= signal.mean()

    # STFT
    f, t, Z = stft(
        signal,
        fs=fs,
        nperseg=int(WINDOW_SECONDS * fs),
        noverlap=int(WINDOW_SECONDS * fs * OVERLAP),
        nfft=NFFT
    )

    P = np.abs(Z) ** 2

    # Doppler range
    mask = f <= MAX_FREQ
    f = f[mask]
    P = P[mask, :]

    # log scale for contrast
    P = 10 * np.log10(P + 1e-12)

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10, 4))
    plt.imshow(
        P,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], f[0], f[-1]],
        cmap="jet"
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Doppler Frequency (Hz)")
    plt.title("CSI Doppler Heatmap")
    plt.tight_layout()
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pkl_file", help="Path to CSI .pkl file")
    args = p.parse_args()

    plot_doppler_heatmap(args.pkl_file)
