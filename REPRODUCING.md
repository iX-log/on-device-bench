# Reproducing

Setup and full step-by-step instructions for reproducing every number in
[RESULTS.md](RESULTS.md), starting from a bare checkout. See
[README.md](README.md) for what this project is and why.

## Setup

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

## Reproducing

End to end, from a bare checkout to a number in RESULTS.md. Steps 1-3 and
8-9 are scripted; step 4 is scripted too if you used the `fetch-models.sh`
shortcut in step 1 (it places the models for Xcode as well as for the
Python side). Steps 5-6 happen by hand in Xcode and on the device; that
hand-off isn't automated.

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

   Shortcut: `scripts/fetch-models.sh` downloads the same three
   `.mlpackage` directories from the project's GitHub release instead of
   converting them yourself, and also copies them into
   `ios/BenchApp/BenchApp/` so Xcode picks them up without step 4 below.

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

4. **Get the model into Xcode.** If you used the `fetch-models.sh`
   shortcut in step 1, this is already done and you can skip to step 5.

   If you converted locally instead: the three `.mlpackage` directories
   are gitignored and aren't referenced by path anywhere in the Xcode
   project. `BenchApp`'s group in Xcode is a file-system-synchronized
   folder rooted at `ios/BenchApp/BenchApp/` (Xcode 16+), meaning
   anything placed in that folder on disk is picked up automatically, no
   explicit "add to target" step required. Copy each of
   `models/whisper-base-encoder-{fp16,int8,int4}.mlpackage` into
   `ios/BenchApp/BenchApp/` (Finder, or
   `cp -R models/whisper-base-encoder-*.mlpackage ios/BenchApp/BenchApp/`).
   Xcode compiles them to `.mlmodelc` and generates the Swift model
   classes (`whisper_base_encoder_fp16`, etc.) that
   `BenchmarkViewModel.swift` calls directly in `loadEncoder(precision:)`.
   If a class name doesn't match, the build fails there, not at
   runtime.

5. **Build and run on the physical device.** Open
   `ios/BenchApp/BenchApp.xcodeproj`. The bundle identifier
   (`com.ixdev.benchapp`) and signing team are tied to the original
   author's Apple Developer account, so on anyone else's machine the
   build will fail to sign. In the target's Signing & Capabilities tab,
   change the bundle identifier to something unique (e.g.
   `com.<yourname>.benchapp`) and select your own team, then select the
   device (not a simulator) as the run destination, build and run. Then
   match the standard conditions used for every number in RESULTS.md:
   airplane mode on, off charger, cooled 5 minutes before the run,
   launched fresh from the home screen (not resumed from Xcode), no
   debugger attached.

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
      chart in sessions 3, 4, and 8. If the device isn't in a nominal
      thermal state when you start it, the app now warns and asks for
      confirmation before proceeding, so expect that prompt rather than
      treating it as a bug.
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

   `BUNDLE` must match whatever you set as the bundle identifier in step
   5, not the default here; if you changed it there and forget to
   override it here, this script will fail to find the app's data
   container on the device. Both are overridable via environment
   variables. Find your device's UDID with `xcrun devicectl list
   devices` and run:

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
