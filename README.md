# on-device-bench

What actually runs on an iPhone under Core ML, measured on physical
hardware — not estimated from a spec sheet or timed in the simulator. Reproducible
benchmarks for on-device model inference — latency, memory, thermal
throttling, and accuracy across quantization levels — using the
Whisper-base encoder as the running example.

**Scope:**
- Core ML specifically, not on-device inference in general — there's no
  comparison here to ONNX Runtime, TFLite, or whisper.cpp on the same
  hardware.
- Encoder only — the decoder never ran on device, so there is no
  end-to-end transcription figure here. A real speech-to-text app would
  also pay decoder cost.

| Measurement | Result | Session |
|---|---|---|
| Steady-state inference | 43.0ms median, 44.2ms p95 (100 runs) | 6 |
| Load time, cold vs. warm | 2046ms → 135ms | 2 |
| Sustained-run behavior | Flat ~41.7ms until ~102s, then a near-vertical step to a ~49ms plateau — a cliff, not a slope | 8 |
| Quantization vs. speed | int4 is 4x smaller than fp16 but only ~3% faster (compute-bound, not memory-bound) | 5 |
| Quantization vs. accuracy | WER 3.4% (fp16) → 3.8% (int8) → 8.8% (int4) | 7 |
| Memory ceiling before jetsam kill | ~3060MB on a 6GB device — half of RAM, not most of it | 9 |

![Overlay of two sustained runs showing latency flat at ~41.7ms then stepping near-vertically to a plateau at ~102 seconds on both battery and power](results/charts/sustained_comparison.png)

![Bar chart of disk size against a line of aggregate WER across fp16, int8, and int4, showing int8 barely raises error while int4 nearly triples it for a further 2x size reduction](results/charts/quantization-tradeoff.png)

Full conditions, caveats, and raw data behind every number above:
[RESULTS.md](RESULTS.md).

## Status
The encoder's latency, memory, thermal, and quantization behavior are
characterized (see the table above); the decoder and any cross-runtime
comparison are not yet covered. Next planned: per-precision layer tracing
via the Core ML Instruments template, to check which compute unit
(ANE/GPU/CPU) each op actually runs on — see [IDEAS.md](IDEAS.md). Full
session-by-session log in [RESULTS.md](RESULTS.md).

## Setup

- Device under test: iPhone 14 Pro Max (A16, 6GB), iOS 27
- Build machine: a Mac (this project used a MacBook M4)
- Xcode with the iOS 27 SDK, and a physical device to run on — the
  simulator doesn't have an ANE, so latency and thermal numbers won't
  reproduce there
- Python 3.12

The Python side has no `requirements.txt` yet — install directly:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch coremltools transformers numpy fsspec pyarrow soundfile jiwer pandas matplotlib
```

(This was run against coremltools 9.0, torch 2.14, transformers 5.16 —
if a fresh install breaks, that version drift is the first thing to check.)

## Reproducing

End to end, from a bare checkout to a number in RESULTS.md. Steps 1-3 and
8-9 are scripted; steps 4-6 are done by hand in Xcode and on the device —
that hand-off is a real gap in the pipeline, not an oversight, and is
called out where it matters.

1. **Convert the model** (Mac). Traces the HF Whisper encoder, converts to
   Core ML, and for int8/int4 applies post-training quantization, saving
   to `models/` (gitignored — not committed):

   ```
   python convert/convert_whisper.py --precision fp16
   python convert/convert_whisper.py --precision int8
   python convert/convert_whisper.py --precision int4
   ```

   Each run prints a numeric verification against the PyTorch original
   (max/mean abs diff, max rel diff) — this is session 0's method.

2. **Prepare audio fixtures** (Mac). Pulls LibriSpeech test-clean
   utterances from the Hub and runs them through Whisper's feature
   extractor:

   ```
   python convert/prepare_audio.py --n 20            # unpacked, for the real-input latency check (session 6)
   python convert/prepare_audio.py --n 20 --pack      # packed into ~30s windows, for WER (session 7)
   ```

3. **Export to raw `.bin` for the iOS app** (Mac) — the step done by hand
   in this project, but scripted here:

   ```
   python convert/export_mel_bin.py --n 5
   python convert/export_mel_bin_packed.py
   ```

   Writes flat little-endian float32 `.bin` files (no header) into
   `ios/BenchApp/BenchApp/mel/` and `mel-packed/`, which is what
   `BenchApp` `memcpy`s straight into an `MLMultiArray`. These `.bin`
   mel fixtures are already committed to git, so this step is only
   needed if you regenerate the fixtures from scratch — unlike the
   `.mlpackage` models from step 1, which are gitignored and never
   committed.

4. **Get the model into Xcode — manual, not scriptable.** The three
   `.mlpackage` directories are gitignored and aren't referenced by path
   anywhere in the Xcode project. Drag each of
   `models/whisper-base-encoder-{fp16,int8,int4}.mlpackage` into the
   BenchApp target in Xcode ("Copy items if needed", add to target).
   Xcode compiles them to `.mlmodelc` and generates the Swift model
   classes (`whisper_base_encoder_fp16`, etc.) that
   `BenchmarkViewModel.swift` calls directly in `loadEncoder(precision:)`
   — if a class name doesn't match, the build fails there, not at
   runtime.

5. **Build and run on the physical device.** Open
   `ios/BenchApp/BenchApp.xcodeproj`, select the device (not a
   simulator) as the run destination, build and run. Then match the
   standard conditions used for every number in RESULTS.md: airplane
   mode on, off charger, cooled 5 minutes before the run, launched fresh
   from the home screen (not resumed from Xcode), no debugger attached.

6. **Run the measurement protocol.** This is the actual method behind
   every number in RESULTS.md, not an aside — pick a precision and
   input mode in the app, then run in this order:

   1. Toggle **Real input** off (synthetic) or on (LibriSpeech mel) —
      session 6 established these are equivalent for timing.
   2. **Quick (100)** — 100 inferences, run 1 discarded as warm-up per
      session 1's convention; reports load time, median/p95/min/max, and
      peak memory footprint (sessions 1, 5, 6).
   3. **Sustained (10 min)** — runs continuously for 600s, sampling
      latency and `thermalState` throughout, and writes a JSON file to
      the app's Documents directory (sessions 3, 4, 8).
   4. **Dump features** — runs the packed 30s windows through the
      encoder and writes each output tensor to Documents as
      `window-N-{precision}.bin`, for WER scoring in step 8 (session 7).
   5. **Memory ceiling** — allocates 32MB blocks until jetsam kills the
      process, fsyncing progress after every block since there's no
      on-kill callback to catch the final state (session 9).

7. **Pull results off the device** (Mac):

   ```
   scripts/pull-results.sh
   ```

   **Flag — this only works on the original machine as written.** The
   script hardcodes one device UDID and bundle ID:

   ```
   DEVICE="${DEVICE:-00008120-0002695E3A68C01E}"   # Ixhen's iPhone 14 Pro Max
   BUNDLE="${BUNDLE:-com.ixdev.benchapp}"
   ```

   Both are overridable via environment variables — find your device's
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

   **Flag — two more spots that assume this project's specific runs:**
   `plot_sustained.py`'s `RUN_LABELS` dict matches charts to filenames by
   timestamp (`sustained-1788785293` → "on-power", etc.) — a new
   sustained run gets a chart but not an "on-power"/"battery" label
   until you add its filename there. `plot_wer.py`'s WER and disk-size
   numbers are hardcoded from RESULTS.md sessions 5 and 7, not read from
   `score_wer.py`'s output — rerunning step 8 doesn't update the chart
   until those two dicts are edited by hand.

## Limitations

What the numbers above can and can't be used for:

- **Encoder only — the decoder never ran on-device.** Every latency,
  memory, and thermal figure here covers the encoder alone; the decoder
  ran on a Mac CPU throughout. There is no end-to-end on-device
  transcription number anywhere in this project.
- **n=1 device.** Every measurement comes from one specific iPhone 14 Pro
  Max unit. Battery health, unit-to-unit ANE variance, and other-model
  behavior are all unmeasured.
- **One model at ~20M params, encoder only** — and the "compute-bound,
  not memory-bandwidth-bound" explanation for the quantization-vs-speed
  finding is inferred from latency numbers, not confirmed with a layer
  trace showing which compute unit ran what.
- **Core ML only.** Nothing here compares against ONNX Runtime, TFLite,
  or whisper.cpp on the same hardware — these numbers describe this
  stack, not on-device inference in general.
- **WER is from 7 windows built out of 20 utterances.** A small sample
  where one window alone swung the aggregate (session 7); not a
  substitute for a proper eval set.
- **The jetsam ceiling is one scenario:** a foreground app aggressively
  allocating with nothing else running and no response to memory
  warnings. Treat ~3060MB as an optimistic upper bound, not a general
  budget (session 9).
- **Raw per-run data is committed for sessions 3, 4, and 9 only.**
  Sessions 1, 2, 5, 6, and 7 exist only as summary statistics — there's
  no way to re-derive their underlying distributions.
- **No repeat-session variance check except the memory ceiling** (three
  runs, agreed within 0.1MB). Every other figure is a single session's
  output, with nothing showing it's stable run to run.

Full per-session caveats — what was measured under non-standard
conditions, what's still unexplained, what a session itself flagged as
needing a rerun — are in [RESULTS.md](RESULTS.md).

## License

Code is [MIT](LICENSE). Measurement data under `results/` — raw JSON,
charts, screenshots — is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/):
cite this repo if you use the numbers.

## Author

Ixhen Hasani — [ix-dev.com](https://ix-dev.com)

Cross-device data is this project's biggest gap — everything here is one
iPhone 14 Pro Max (A16, 6GB). If a number here doesn't reproduce on your
own run, or you run this on other hardware (an A17 or A18 result would
be genuinely useful), please open an issue with your conditions and
output.
