"""Prepare a small LibriSpeech test-clean sample for accuracy testing.

Pulls a handful of utterances directly from the openslr/librispeech_asr
parquet mirror on the Hub (one row group over HTTP range requests, so this
does not pull the full ~350MB test-clean file), decodes the embedded FLAC
audio, and runs each through WhisperFeatureExtractor to get the same
log-mel spectrogram the Core ML encoder expects (80 mels x 3000 frames,
fixed 30s window -- see convert_whisper.py).

Saves one .npy per utterance plus a manifest.json mapping filenames to
reference transcripts, for word-error-rate comparisons later.

With --pack, consecutive utterances are concatenated (separated by a short
silence gap) into ~30s windows instead of being padded individually -- most
LibriSpeech utterances are a few seconds long, so an unpacked 30s window is
mostly Whisper's own zero-padding. Packing trades one .npy per utterance for
one .npy per window, each holding several utterances' worth of real audio.
"""

import argparse
import io
import json
import os
import statistics

import fsspec
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from transformers import WhisperFeatureExtractor

PARQUET_URL = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/"
    "resolve/main/all/test.clean/0000.parquet"
)
WHISPER_WINDOW_S = 30.0


def fetch_rows(n):
    print(f"reading row group from {PARQUET_URL} ...")
    with fsspec.open(PARQUET_URL, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(0)
    rows = table.slice(0, n).to_pylist()
    print(f"got {len(rows)} utterances")
    return rows


def load_utterances(rows):
    """Decode each row's audio once, up front, so both modes can share it."""
    utterances = []
    sr = None
    for row in rows:
        audio, row_sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if sr is None:
            sr = row_sr
        elif row_sr != sr:
            raise ValueError(f"sample rate mismatch: {row['id']} is {row_sr}Hz, expected {sr}Hz")
        utterances.append({
            "id": row["id"],
            "transcript": row["text"],
            "audio": audio,
            "duration_s": len(audio) / row_sr,
        })
    return utterances, sr


def pack_windows(utterances, sr, gap_s):
    """Group consecutive utterances into windows of at most WHISPER_WINDOW_S,
    each utterance separated by gap_s of silence. Never splits an utterance."""
    gap = np.zeros(int(gap_s * sr), dtype=utterances[0]["audio"].dtype)

    windows = []
    buf = []
    buf_duration = 0.0
    for u in utterances:
        addition = u["duration_s"] + (gap_s if buf else 0.0)
        if buf and buf_duration + addition > WHISPER_WINDOW_S:
            windows.append(buf)
            buf, buf_duration, addition = [], 0.0, u["duration_s"]
        buf.append(u)
        buf_duration += addition
    if buf:
        windows.append(buf)

    packed = []
    for w in windows:
        parts = []
        for i, u in enumerate(w):
            if i > 0:
                parts.append(gap)
            parts.append(u["audio"])
        audio = np.concatenate(parts)
        packed.append({
            "ids": [u["id"] for u in w],
            "transcript": " ".join(u["transcript"] for u in w),
            "audio": audio,
            "duration_s": len(audio) / sr,
        })
    return packed


def extract_and_save(items, sr, extractor, mel_dir, packed):
    """Run the feature extractor over each item's audio and save the mel.
    Returns (manifest dict, list of durations, first mel array)."""
    manifest = {}
    durations = []
    first_mel = None
    for i, item in enumerate(items):
        features = extractor(item["audio"], sampling_rate=sr, return_tensors="np")
        mel = features.input_features[0].astype(np.float32)  # (80, 3000)

        filename = f"window{i:03d}.npy" if packed else f"{item['id']}.npy"
        np.save(os.path.join(mel_dir, filename), mel)

        entry = {"transcript": item["transcript"], "duration_s": round(item["duration_s"], 3)}
        if "ids" in item:
            entry["utterance_ids"] = item["ids"]
        manifest[filename] = entry

        durations.append(item["duration_s"])
        if first_mel is None:
            first_mel = mel
    return manifest, durations, first_mel


def print_duration_summary(durations):
    real_fraction = statistics.mean(min(d, WHISPER_WINDOW_S) / WHISPER_WINDOW_S for d in durations)
    print(
        f"duration (s):  min {min(durations):.2f}  "
        f"median {statistics.median(durations):.2f}  "
        f"max {max(durations):.2f}"
    )
    print(
        f"of the {WHISPER_WINDOW_S:.0f}s window: "
        f"{real_fraction:.1%} real audio, {1 - real_fraction:.1%} padding (avg)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/librispeech-test-clean")
    ap.add_argument("--pack", action="store_true", help="pack utterances into ~30s windows")
    ap.add_argument("--gap", type=float, default=0.2, help="silence gap between packed utterances, seconds")
    args = ap.parse_args()

    mel_dir = os.path.join(args.out, "mel")
    os.makedirs(mel_dir, exist_ok=True)

    extractor = WhisperFeatureExtractor.from_pretrained(args.model)

    rows = fetch_rows(args.n)
    utterances, sr = load_utterances(rows)

    if args.pack:
        items = pack_windows(utterances, sr, args.gap)
        print(f"packed {len(utterances)} utterances -> {len(items)} windows (gap {args.gap}s)")
    else:
        items = utterances

    manifest, durations, first_mel = extract_and_save(items, sr, extractor, mel_dir, args.pack)

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved {len(manifest)} spectrograms -> {mel_dir}")
    print(f"manifest -> {manifest_path}")
    print(f"first spectrogram shape: {first_mel.shape}")
    print(f"first spectrogram range: [{first_mel.min():.4f}, {first_mel.max():.4f}]")

    print_duration_summary(durations)


if __name__ == "__main__":
    main()
