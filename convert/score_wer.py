"""Score WER on transcripts decoded from on-device encoder output.

Loads each window-N-{precision}.bin dumped by BenchApp's "Dump features" mode
(results/device-pull/, via convert/prepare_audio.py --pack's packed windows),
reshapes it back to the encoder's (1, 1500, 512) float32 output, and feeds it
to WhisperForConditionalGeneration.generate() as encoder_outputs -- skipping
the encoder entirely, since that's the part we already ran on-device. Only
the decoder runs here, on CPU.

Reference and hypothesis are both normalized before scoring (lowercase,
expand contractions, strip punctuation, collapse whitespace) via jiwer's
transform pipeline, so wording differences that don't affect meaning --
"don't" vs "do not", a missing comma -- don't inflate the error rate.
"""

import argparse
import glob
import json
import os
import re

import jiwer
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput

ENCODER_SHAPE = (1, 1500, 512)
FILENAME_RE = re.compile(r"window-(\d+)-(\w+)\.bin$")

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def find_windows(bin_dir, precision):
    windows = []
    for path in glob.glob(os.path.join(bin_dir, f"window-*-{precision}.bin")):
        m = FILENAME_RE.search(path)
        if m:
            windows.append((int(m.group(1)), path))
    return sorted(windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="results/device-pull")
    ap.add_argument("--manifest", default="data/librispeech-test-clean-packed/manifest.json")
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--verbose", action="store_true", help="print each window's transcripts side by side")
    args = ap.parse_args()

    windows = find_windows(args.bin_dir, args.precision)
    if not windows:
        raise SystemExit(f"no window-*-{args.precision}.bin files found in {args.bin_dir}")

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"loading {args.model} ...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    tokenizer = WhisperTokenizer.from_pretrained(args.model)

    references = []
    hypotheses = []

    for index, path in windows:
        manifest_key = f"window{index:03d}.npy"
        reference = manifest.get(manifest_key, {}).get("transcript", "<no reference found>")

        hidden = np.fromfile(path, dtype="<f4")
        if hidden.size != np.prod(ENCODER_SHAPE):
            raise ValueError(f"{path}: expected {np.prod(ENCODER_SHAPE)} floats, got {hidden.size}")
        hidden = torch.from_numpy(hidden.reshape(ENCODER_SHAPE))

        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
        with torch.no_grad():
            ids = model.generate(encoder_outputs=encoder_outputs, language="en", task="transcribe")
        hypothesis = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

        references.append(reference)
        hypotheses.append(hypothesis)

        window_wer = jiwer.wer(
            reference, hypothesis, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE
        )

        if args.verbose:
            print(f"\n--- window {index} ({os.path.basename(path)}) ---")
            print(f"ref: {reference}")
            print(f"hyp: {hypothesis}")
        print(f"window {index}: WER {window_wer:.1%}")

    aggregate_wer = jiwer.wer(
        references, hypotheses, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE
    )
    print(f"\naggregate WER ({len(windows)} windows, over total words): {aggregate_wer:.1%}")


if __name__ == "__main__":
    main()
