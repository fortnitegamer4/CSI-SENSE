import os
import pickle
import argparse
import numpy as np
from glob import glob
from CSIKit.reader import get_reader


def convert_one(pcap_path, out_path=None):
    if out_path is None:
        out_path = os.path.splitext(pcap_path)[0] + ".pkl"

    reader = get_reader(pcap_path)
    csi_data = reader.read_file(pcap_path)

    frames = []
    timestamps = []

    for frame in csi_data.frames:
        try:
            csi = np.array(frame.csi_matrix)
            csi = np.squeeze(csi)

            if csi.ndim > 1:
                csi = csi.reshape(-1)

            if csi.size == 0:
                continue

            real_ok = np.all(np.isfinite(np.real(csi)))
            imag_ok = np.all(np.isfinite(np.imag(csi)))
            if not (real_ok and imag_ok):
                continue

            frames.append(csi)
            timestamps.append(frame.timestamp)

        except Exception:
            continue

    if not frames:
        print(f"[SKIP] no valid CSI frames: {pcap_path}")
        return None

    min_len = min(len(f) for f in frames)
    frames = [f[:min_len] for f in frames]

    csi_mat = np.stack(frames, axis=0).T
    ts = np.array(timestamps[:len(frames)], dtype=float)

    out = {
        "valid_csi": csi_mat,
        "valid_csi_timestmp_array": ts,
    }

    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"[OK] {pcap_path} -> {out_path}")
    print(f"     valid_csi shape: {csi_mat.shape}")
    print(f"     timestamps shape: {ts.shape}")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="one .pcap file or a folder containing .pcap files")
    args = ap.parse_args()

    path = args.path

    if os.path.isfile(path):
        convert_one(path)
        return

    if os.path.isdir(path):
        pcaps = sorted(glob(os.path.join(path, "*.pcap")))
        if not pcaps:
            print(f"[!] no .pcap files found in {path}")
            return

        for pcap in pcaps:
            convert_one(pcap)
        return

    raise FileNotFoundError(path)


if __name__ == "__main__":
    main()