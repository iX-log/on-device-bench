"""Export a handful of mel spectrograms as raw .bin files for the iOS app.

Takes the first N .npy files from prepare_audio.py's output (80 x 3000
float32, matching the encoder's fixed 30s input) and writes each as a flat
little-endian float32 .bin -- no header, no shape info -- so BenchApp can
load it straight into an MLMultiArray with memcpy.
"""

import argparse
import glob
import os

import numpy as np

EXPECTED_SHAPE = (80, 3000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/librispeech-test-clean/mel")
    ap.add_argument("--out", default="ios/BenchApp/BenchApp/mel")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    npy_files = sorted(glob.glob(os.path.join(args.src, "*.npy")))[: args.n]
    if not npy_files:
        raise SystemExit(f"no .npy files found in {args.src}")

    os.makedirs(args.out, exist_ok=True)

    for path in npy_files:
        mel = np.load(path)
        if mel.shape != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected shape {EXPECTED_SHAPE}, got {mel.shape}")

        flat = mel.astype("<f4", copy=False)
        name = os.path.splitext(os.path.basename(path))[0] + ".bin"
        out_path = os.path.join(args.out, name)
        flat.tofile(out_path)
        print(f"{out_path}  ({flat.nbytes} bytes)")

    print(f"wrote {len(npy_files)} .bin files -> {args.out}")


if __name__ == "__main__":
    main()
