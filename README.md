# you-get-100-seconds

*[Attention is all you need](https://arxiv.org/abs/1706.03762). 100 seconds is all you get.*

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Platform: Core ML / iOS](https://img.shields.io/badge/platform-Core%20ML%20%2F%20iOS-black.svg)](#setup)

Before you put an AI model on an iPhone, you want to know:

- **Speed**: how fast does it actually run?
- **Memory**: how much RAM does it really use?
- **Endurance**: does it get slower the longer you run it?
- **Accuracy**: what do you lose by shrinking the model?

This repo answers all four, measured on physical hardware, across
quantization levels, using the Whisper-base encoder as the running
example.

<details>
<summary><h2>Setup</h2></summary>

- Device under test: iPhone 14 Pro Max (A16, 6GB), iOS 27
- Build machine: a Mac (this project used a MacBook M4)
- Xcode with the iOS 27 SDK, and a physical device to run on (the
  simulator doesn't have an ANE, so latency and thermal numbers won't
  reproduce there)
- Python 3.12

The Python side has no `requirements.txt` yet. Install directly:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch coremltools transformers numpy fsspec pyarrow soundfile jiwer pandas matplotlib
```

(This was run against coremltools 9.0, torch 2.14, transformers 5.16.
If a fresh install breaks, that version drift is the first thing to check.)

</details>

<details>
<summary><h2>Key Findings</h2></summary>

| Measurement | Result | Session |
|---|---|---|
| Steady-state inference | 43.0ms median, 44.2ms p95 (100 runs) | 6 |
| Load time, cold vs. warm | 2046ms → 135ms | 2 |
| Sustained-run behavior | Flat ~41.7ms until ~102s, then a near-vertical step to a ~49ms plateau (a cliff, not a slope) | 8 |
| Quantization vs. speed | int4 is 4x smaller than fp16 but only ~3% faster (compute-bound, not memory-bound) | 5 |
| Quantization vs. accuracy | WER 3.4% (fp16) → 3.8% (int8) → 8.8% (int4) | 7 |
| Memory ceiling before jetsam kill | ~3060MB on a 6GB device (half of RAM, not most of it) | 9 |

Conditions, caveats, and raw data: [RESULTS.md](RESULTS.md).

</details>

<details>
<summary><h2>Charts</h2></summary>

![Overlay of two sustained runs showing latency flat at ~41.7ms then stepping near-vertically to a plateau at ~102 seconds on both battery and power](results/charts/sustained_comparison.png)

![Bar chart of disk size against a line of aggregate WER across fp16, int8, and int4, showing int8 barely raises error while int4 nearly triples it for a further 2x size reduction](results/charts/quantization-tradeoff.png)

</details>

<details>
<summary><h2>Status</h2></summary>

Decoder behavior and cross-runtime comparison aren't covered yet. Next
planned: per-precision layer tracing
via the Core ML Instruments template, to check which compute unit
(ANE/GPU/CPU) each op actually runs on. See [IDEAS.md](IDEAS.md). Full
session-by-session log in [RESULTS.md](RESULTS.md).

</details>

<details>
<summary><h2>Reproducing</h2></summary>

End to end, from a bare checkout to a number in RESULTS.md. Steps 1-3 and
8-9 are scripted; steps 4-6 happen by hand in Xcode and on the device.
That hand-off isn't automated yet.

1. **Convert the model** (Mac). Traces the HF Whisper encoder, converts to
   Core ML, and for int8/int4 applies post-training quantization, saving
   to `models/` (gitignored, not committed):

   ```
   python convert/convert_whisper.py --precision fp16
   python convert/convert_whisper.py --precision int8
   python convert/convert_whisper.py --precision int4
   ```

   Each run prints a numeric verification against the PyTorch original
   (max/mean abs diff, max rel diff). This is session 0's method.

2. **Prepare audio fixtures** (Mac). Pulls LibriSpeech test-clean
   utterances from the Hub and runs them through Whisper's feature
   extractor:

   ```
   python convert/prepare_audio.py --n 20            # unpacked, for the real-input latency check (session 6)
   python convert/prepare_audio.py --n 20 --pack      # packed into ~30s windows, for WER (session 7)
   ```

3. **Export to raw `.bin` for the iOS app** (Mac). The step done by hand
   in this project, but scripted here:

   ```
   python convert/export_mel_bin.py --n 5
   python convert/export_mel_bin_packed.py
   ```

   Writes flat little-endian float32 `.bin` files (no header) into
   `ios/BenchApp/BenchApp/mel/` and `mel-packed/`, which is what
   `BenchApp` `memcpy`s straight into an `MLMultiArray`. These `.bin`
   mel fixtures are already committed to git, so this step is only
   needed if you regenerate the fixtures from scratch, unlike the
   `.mlpackage` models from step 1, which are gitignored and never
   committed.

4. **Get the model into Xcode (manual, not scriptable).** The three
   `.mlpackage` directories are gitignored and aren't referenced by path
   anywhere in the Xcode project. Drag each of
   `models/whisper-base-encoder-{fp16,int8,int4}.mlpackage` into the
   BenchApp target in Xcode ("Copy items if needed", add to target).
   Xcode compiles them to `.mlmodelc` and generates the Swift model
   classes (`whisper_base_encoder_fp16`, etc.) that
   `BenchmarkViewModel.swift` calls directly in `loadEncoder(precision:)`.
   If a class name doesn't match, the build fails there, not at
   runtime.

5. **Build and run on the physical device.** Open
   `ios/BenchApp/BenchApp.xcodeproj`, select the device (not a
   simulator) as the run destination, build and run. Then match the
   standard conditions used for every number in RESULTS.md: airplane
   mode on, off charger, cooled 5 minutes before the run, launched fresh
   from the home screen (not resumed from Xcode), no debugger attached.

6. **Run the measurement protocol.** Pick a precision and input mode in
   the app, then run in this order. This is the method behind every
   number in RESULTS.md:

   1. Toggle **Real input** off (synthetic) or on (LibriSpeech mel).
      Session 6 found these equivalent for timing, so either is fine.
   2. **Quick (100)** runs 100 inferences (the first discarded as
      warm-up, session 1's convention) and reports load time,
      median/p95/min/max, and peak memory footprint.
   3. **Sustained (10 min)** runs continuously for 600s, sampling
      latency and `thermalState` throughout, and writes a JSON file to
      the app's Documents directory. This is what produced the cliff
      chart in sessions 3, 4, and 8.
   4. **Dump features** runs the packed 30s windows through the encoder
      and writes each output tensor to Documents as
      `window-N-{precision}.bin`. Session 7 used this for WER scoring
      in step 8 below.
   5. **Memory ceiling** allocates 32MB blocks until jetsam kills the
      process, fsyncing progress after every block since there's no
      on-kill callback to catch the final state. See session 9.

7. **Pull results off the device** (Mac):

   ```
   scripts/pull-results.sh
   ```

   This only works on the original machine as written. The script
   hardcodes one device UDID and bundle ID:

   ```
   DEVICE="${DEVICE:-00008120-0002695E3A68C01E}"   # Ixhen's iPhone 14 Pro Max
   BUNDLE="${BUNDLE:-com.ixdev.benchapp}"
   ```

   Both are overridable via environment variables. Find your device's
   UDID with `xcrun devicectl list devices` and run:

   ```
   DEVICE=<your-udid> BUNDLE=<your-bundle-id> scripts/pull-results.sh
   ```

8. **Score accuracy** (Mac), after a "Dump features" run has been pulled:

   ```
   python convert/score_wer.py --precision fp16
   python convert/score_wer.py --precision int8
   python convert/score_wer.py --precision int4
   ```

   Feeds each pulled encoder output into the full-precision PyTorch
   decoder and scores WER against the manifest transcripts with `jiwer`.

9. **Regenerate the charts** (Mac):

   ```
   python analysis/plot_sustained.py
   python analysis/plot_wer.py
   ```

   Two more spots that assume this project's specific runs:
   `plot_sustained.py`'s `RUN_LABELS` dict matches charts to filenames by
   timestamp (`sustained-1788785293` → "on-power", etc.). A new
   sustained run gets a chart but not an "on-power"/"battery" label
   until you add its filename there. `plot_wer.py`'s WER and disk-size
   numbers are hardcoded from RESULTS.md sessions 5 and 7, not read from
   `score_wer.py`'s output. Rerunning step 8 doesn't update the chart
   until those two dicts are edited by hand.

</details>

<details>
<summary><h2>Scope</h2></summary>

Core ML only, no comparison to ONNX Runtime, TFLite, or whisper.cpp on
the same hardware. And encoder only: the decoder never ran on device, so
there's no end-to-end transcription number in this repo, and a real
speech-to-text app would still pay decoder cost on top of everything
measured here.

</details>

<details>
<summary><h2>Limitations</h2></summary>

- Every number here is the encoder alone. The decoder ran on a Mac CPU the
  whole time, so there's no end-to-end on-device transcription figure
  anywhere in this project. A real app would pay decoder cost on top.
- **n=1 device.** One specific iPhone 14 Pro Max, one unit. Battery
  health, unit-to-unit ANE variance, and how other model architectures
  behave are all unmeasured.
- The "compute-bound, not memory-bandwidth-bound" read on the
  quantization-vs-speed result is inferred from latency numbers alone.
  Nobody's traced which compute unit actually ran which op.
- These numbers describe this one stack. No comparison to ONNX Runtime,
  TFLite, or whisper.cpp anywhere here.
- WER comes from 7 windows built out of 20 utterances. In session 7, one
  window alone swung the aggregate. This is not a proper eval set.
- The jetsam ceiling reflects one scenario: a foreground app aggressively
  allocating memory, nothing else running, no response to memory
  warnings. ~3060MB is closer to an optimistic upper bound than a budget
  you should plan around.
- Raw per-run data only exists for sessions 3, 4, and 9. Sessions 1, 2, 5,
  6, and 7 are summary statistics only. Their underlying distributions
  are gone.
- Nothing got a repeat-session variance check except the memory ceiling
  (three runs, agreed within 0.1MB). Every other number is from a single
  session.

Full per-session caveats (what was measured under non-standard
conditions, what's still unexplained, what a session itself flagged as
needing a rerun) are in [RESULTS.md](RESULTS.md).

</details>

<details>
<summary><h2>License</h2></summary>

Code is [MIT](LICENSE). Measurement data under `results/` (raw JSON,
charts, screenshots) is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/):
cite this repo if you use the numbers.

</details>

<details>
<summary><h2>Author</h2></summary>

Ixhen Hasani, [ix-dev.com](https://ix-dev.com)

Cross-device data is this project's biggest gap: everything here is one
iPhone 14 Pro Max (A16, 6GB). If a number here doesn't reproduce on your
own run, or you run this on other hardware (an A17 or A18 result would
be genuinely useful), please open an issue with your conditions and
output.

</details>
