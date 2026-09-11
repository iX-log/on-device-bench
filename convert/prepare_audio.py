"""Prepare a small LibriSpeech test-clean sample for accuracy testing.

Pulls a handful of utterances directly from the openslr/librispeech_asr
parquet mirror on the Hub (one row group over HTTP range requests, so this
does not pull the full ~350MB test-clean file), decodes the embedded FLAC
audio, and runs each through WhisperFeatureExtractor to get the same
log-mel spectrogram the Core ML encoder expects (80 mels x 3000 frames,
fixed 30s window -- see convert_whisper.py).

Saves one .npy per utterance plus a manifest.json mapping filenames to
reference transcripts, for word-error-rate comparisons later.
"""

import argparse
import io
import json
import os

import fsspec
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperFeatureExtractor

PARQUET_URL = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/"
    "resolve/main/all/test.clean/0000.parquet"
)


def fetch_rows(n):
    print(f"reading row group from {PARQUET_URL} ...")
    with fsspec.open(PARQUET_URL, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(0)
    rows = table.slice(0, n).to_pylist()
    print(f"got {len(rows)} utterances")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/librispeech-test-clean")
    args = ap.parse_args()

    mel_dir = os.path.join(args.out, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    extractor = WhisperFeatureExtractor.from_pretrained(args.model)

    rows = fetch_rows(args.n)

    manifest = {}
    first_mel = None
    for row in rows:
        utt_id = row["id"]
        transcript = row["text"]

        audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        features = extractor(audio, sampling_rate=sr, return_tensors="np")
        mel = features.input_features[0].astype(np.float32)  # (80, 3000)

        filename = f"{utt_id}.npy"
        np.save(os.path.join(mel_dir, filename), mel)
        manifest[filename] = transcript

        if first_mel is None:
            first_mel = mel

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved {len(manifest)} spectrograms -> {mel_dir}")
    print(f"manifest -> {manifest_path}")
    print(f"first spectrogram shape: {first_mel.shape}")
    print(f"first spectrogram range: [{first_mel.min():.4f}, {first_mel.max():.4f}]")


if __name__ == "__main__":
    main()
