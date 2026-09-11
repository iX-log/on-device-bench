"""Export packed mel windows as raw .bin files for the iOS app.

Takes the window*.npy files from prepare_audio.py --pack output (80 x 3000
float32, matching the encoder's fixed 30s input) and writes each as a flat
little-endian float32 .bin -- no header, no shape info -- so BenchApp can
load it straight into an MLMultiArray with memcpy. Also copies the packed
manifest.json alongside so the app can look up each window's transcript and
source utterance IDs.
"""

import argparse
import glob
import os
import shutil

import numpy as np

EXPECTED_SHAPE = (80, 3000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/librispeech-test-clean-packed")
    ap.add_argument("--out", default="ios/BenchApp/BenchApp/mel-packed")
    args = ap.parse_args()

    mel_src = os.path.join(args.src, "mel")
    npy_files = sorted(glob.glob(os.path.join(mel_src, "window*.npy")))
    if not npy_files:
        raise SystemExit(f"no window*.npy files found in {mel_src}")

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

    manifest_src = os.path.join(args.src, "manifest.json")
    manifest_dst = os.path.join(args.out, "manifest.json")
    shutil.copyfile(manifest_src, manifest_dst)
    print(f"copied {manifest_src} -> {manifest_dst}")

    print(f"wrote {len(npy_files)} .bin files -> {args.out}")


if __name__ == "__main__":
    main()
