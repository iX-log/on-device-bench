# Results

Measured numbers, with the exact conditions that produced them. If a number
here can't be reproduced under the same conditions, treat it as wrong.

## Device & model

- Device: iPhone 14 Pro Max, A16 Bionic, 6GB RAM, iOS 27.0
- Model: whisper-base encoder, fp16, Core ML, fixed 30s input (`[1, 80, 3000]` mel)

## Standard conditions

Unless a session notes otherwise, every run below was collected with:

- Airplane mode on
- Device off charger (not plugged in)
- Cooled 5 minutes before the run (no back-to-back thermal carryover)
- App launched fresh from the home screen (not resumed from Xcode)
- No debugger attached

## Session 2 — memory instrumentation, cold vs. warm

`phys_footprint` sampled at 10ms intervals during inference; median/p95
computed over runs 2-100 (run 1 discarded as warm-up, per session 1).

| Metric | Cold install | Warm launch |
|---|---|---|
| Load | 1975 ms | 135 ms |
| Model cost (phys_footprint delta after load) | 8.2 MB | 2.9 MB |
| Peak footprint (during inference) | 28.4 MB | 20.6 MB |
| Median inference | 42.1 ms | 46.0 ms |
| p95 inference | 43.1 ms | 47.7 ms |

Notes:

- The `.mlpackage` is 39MB on disk but costs 6-9MB phys_footprint at load —
  Core ML memory-maps weights rather than copying them into the process's
  resident set.
- Warm model cost varies with page cache state: observed anywhere from
  0.1MB to 2.9MB across repeated warm launches, depending on how much of the
  mmap'd weight file the OS still has cached from prior runs.
- Unexplained: warm median (46.0ms) is higher than cold (42.1ms). This is
  backwards from what mmap/page-cache behavior would predict and needs
  rechecking — possibly a confound in how "warm launch" was triggered, or an
  artifact of sample size (100 runs, single session).

## Session 3 — sustained run, thermal drift

Deviates from standard conditions: device was on charger (power), not
airplane-mode-relevant but screen was held on at minimum brightness with the
idle timer disabled for the duration of the run.

12,848 inferences over 600s, whisper-base encoder fp16, iPhone 14 Pro Max.
Raw data: `results/device-pull/sustained-1788785293.json`.

| Metric | Value |
|---|---|
| Load | 2082.9 ms |
| Model cost | 9.3 MB |
| Median (all runs) | 46.5 ms |
| p95 (all runs) | 50.1 ms |
| First-minute median | 41.8 ms |
| Last-minute median | 49.4 ms |
| Drift | +7.7 ms (+18%) |

Thermal state transitions (via `ProcessInfo.thermalState`):

| State | Onset |
|---|---|
| Nominal | 0.1 s |
| Fair | 102.4 s |
| Serious | 347.4 s |

Caveats:

- Run was on power, not battery. Charging heat likely pulls the thermal
  transitions earlier than they'd occur on battery — treat these onset times
  as conservative (i.e. battery-only operation should do at least this well,
  possibly better). A battery rerun is pending.
- The sustained summary does not discard the cold first inference the way
  the quick test does (see session 1). That inflates the first-minute
  median slightly, which means the true drift is understated relative to
  what's reported above.
- Screen was on at minimum brightness with the idle timer disabled for the
  full 600s, which contributes some heat on top of the inference workload
  itself.

### Accidental Low Power Mode measurement

Low Power Mode was on and the device was charging — not a clean isolation
of Low Power Mode alone. Needs a controlled rerun (LPM on/off, off charger,
otherwise standard conditions) before drawing conclusions.

| Metric | LPM + charging | Baseline |
|---|---|---|
| Median | 65.6 ms | 42 ms |
| Median delta | +56% | — |
| p95 | 85.4 ms | — |
| Load | 3376 ms | ~2000 ms |

## Session 4 — sustained run, battery

Same protocol as session 3, but on battery (standard conditions: off
charger) instead of on power.

12,561 inferences over 600s, whisper-base encoder fp16, iPhone 14 Pro Max,
warm launch. Raw data: `results/device-pull/sustained-1788866056.json`.

| Metric | Value |
|---|---|
| Load | 137.9 ms |
| Model cost | 2.8 MB |
| Median (all runs) | 49.2 ms |
| p95 (all runs) | 50.7 ms |
| First-minute median | 41.6 ms |
| Last-minute median | 49.4 ms |
| Drift | +7.8 ms (+19%) |

Thermal state transitions (via `ProcessInfo.thermalState`):

| State | Onset |
|---|---|
| Fair | 56.2 s |
| Serious | 101.2 s |

### Comparison: session 4 (battery) vs. session 3 (power)

| Transition | Battery (session 4) | Power (session 3) |
|---|---|---|
| Fair | 56.2 s | 102.4 s |
| Serious | 101.2 s | 347.4 s |

Serious thermal state arrived 3.4x sooner on battery than on power — the
opposite of what charging heat would predict (session 3's caveats assumed
charging would pull transitions *earlier*, not later). Two candidate
explanations, both untested:

- Low battery level may make iOS more thermally conservative independent of
  actual heat (state-of-charge-driven throttling, not temperature-driven).
- The two runs may have started from different baseline temperatures
  (e.g. device history before the run, ambient conditions), confounding
  the comparison.

Drift held at 18–19% across both runs (+18% on power, +19% on battery),
so the latency degradation itself is reproducible even though the thermal
transition timing is not — the two are not as tightly coupled as assumed.

The battery run also completed fewer inferences in the same 600s window
(12,561 vs. 12,848 on power), consistent with the earlier throttling
observed above.

## Session 5 — precision comparison (fp16 / int8 / int4)

Quick test (100 runs), whisper-base encoder, iPhone 14 Pro Max. Airplane
mode, off power. Run 13:37–13:38.

| Metric | fp16 | int8 | int4 |
|---|---|---|---|
| Disk size | 39.4 MB | 19.8 MB | 10.0 MB |
| Load | 22.2 ms (warm) | 1425.2 ms (cold) | 1369.5 ms (cold) |
| Median inference | 41.8 ms | 41.1 ms | 40.5 ms |
| p95 inference | 43.3 ms | 42.2 ms | 41.2 ms |
| Peak footprint | 26.5 MB | 30.1 MB | 29.3 MB |

Finding: a 4x reduction in file size (fp16 → int4) buys only ~3% latency
improvement (41.8ms → 40.5ms median). At ~20M parameters the encoder is
likely compute-bound rather than memory-bandwidth-bound, so shrinking the
weights relieves no bottleneck. Practical guidance: quantize for storage
and download size, not for speed. Hypothesis to test later: this should
flip for a larger model, where memory bandwidth is more likely to be the
constraint.

Caveats:

- fp16 was measured on a warm load while int8 and int4 were cold — the run
  order was not controlled for this. Rerun all three from the same
  load state (or with cooling/reinstall between each) before trusting the
  load-time column or any cross-precision comparison that touches load.
- "Model cost" (phys_footprint delta at load: 1.6MB fp16, 7.9MB int8,
  6.9MB int4) bears no relation to file size and reflects page cache state
  rather than the model itself (see session 2). Dropped from the table
  above; report peak footprint instead.
- Conversion-time verification against PyTorch gave a max relative error of
  7.3% for int8, which exceeded the arbitrary threshold set in the
  conversion script. That threshold isn't meaningful as a correctness
  signal: the verification input was random noise, not a real spectrogram,
  and max relative difference is a worst-single-element metric that a
  handful of near-zero activations can blow up. Real accuracy numbers need
  word error rate on real audio — that's week 5.

## Session 6 — input validity check (synthetic vs. real)

fp16 only, Quick test (100 runs), iPhone 14 Pro Max. Airplane mode, off
power. Both runs back to back at 10:30.

| Metric | Synthetic (`Float.random`) | Real (LibriSpeech mel, 6930-75918-0000) |
|---|---|---|
| Median | 43.0 ms | 42.9 ms |
| p95 | 44.2 ms | 43.8 ms |
| Min | 42.5 ms | 42.1 ms |
| Max | 44.8 ms | 47.1 ms |
| Peak footprint | 21.0 MB | 23.7 MB |

Finding: input distribution does not affect ANE throughput for this model.
The 0.1ms difference in median is well inside run-to-run variance. This
validates every latency number recorded since session 0 — synthetic input
is a legitimate shortcut for timing work, now verified rather than assumed.

Scope the claim carefully: this holds for timing only. It says nothing
about quantization error, where activation distribution matters a great
deal. The 7.3% max relative error measured for int8 at conversion time
(session 5) used random noise and remains unverified against real audio —
WER will settle that.

Caveats:

- Tested on fp16 only, not int8 or int4.
- The real-input run showed run 1 (cold) at 69.6ms vs. 50.1ms for synthetic
  — likely first-read of the `.bin` from disk rather than inference. It's
  discarded from the statistics either way (see session 1 warm-up
  convention).
- `WhisperFeatureExtractor` pads clips shorter than 30s with silence, so if
  this utterance is short, a large fraction of the benchmarked compute was
  silence. Durations need checking before the WER work.

## Session 7 — quantization accuracy (WER on real audio)

Method: 20 LibriSpeech test-clean utterances packed into 7 windows of
~30s (79.6% real audio, 20.4% padding). Encoder ran on iPhone 14 Pro Max
at each precision; encoder outputs (`1, 1500, 512` float32) dumped to
disk, pulled to Mac, decoded with the full-precision PyTorch Whisper
decoder via `encoder_outputs=`. Scored with `jiwer` after normalization
(lowercase, strip punctuation, collapse whitespace, expand contractions).
Aggregate computed over total words, not averaged per window.

| Precision | Disk size | Median inference (session 5) | Aggregate WER |
|---|---|---|---|
| fp16 | 39.4 MB | 41.8 ms | 3.4% |
| int8 | 19.8 MB | 41.1 ms | 3.8% |
| int4 | 10.0 MB | 40.5 ms | 8.8% |

Per-window WER, fp16: 3.2 / 0.0 / 1.6 / 1.3 / 5.4 / 9.9 / 0.0 %.

Finding: int8 is close to free — half the size for 0.4 points of WER.
int4 is a bad trade: it buys only ~3% on latency (session 5) and costs
2.6x the error rate. On an ANE, 4-bit quantization of this model gives
up substantial accuracy in exchange for download size alone.

Caveats:

- Seven windows is a small sample, and window 5 alone was 9.9% at fp16 —
  the aggregate is sensitive to individual windows. Expand to 50-100
  utterances before publishing.
- Only the encoder is quantized; the decoder ran at full precision in
  PyTorch. A fully quantized pipeline would likely be worse, so these
  numbers are optimistic relative to an on-device encoder+decoder setup.
- Packed multi-speaker windows are not standard LibriSpeech scoring, so
  absolute WER is not comparable to published Whisper figures — only the
  comparison between precisions here is valid.

## Session 8 — charting the sustained runs: cliff, not slope

Charted the two session 3/4 sustained runs from raw JSON.
Charts: `results/charts/sustained-1788785293.png` (power),
`results/charts/sustained-1788866056.png` (battery),
`results/charts/sustained_comparison.png` (overlay).

Revised finding: the thermal degradation is a cliff, not a slope. Latency
holds flat at ~41.7ms until roughly 102 seconds, then steps near-vertically
to a plateau. The session 3/4 framing of "+18% drift over ten minutes" is
arithmetically accurate but misleading — it describes the run as a gradual
decline when the phone actually spends most of the ten minutes flat on a
plateau, with one sharp transition. Practical statement: you get about
100 seconds at full speed, then you're on a different machine.

Both runs step at almost exactly 102s despite different power conditions
(battery vs. on-charger), so the cliff location is a property of the
thermal envelope, not the power condition — a conclusion more robust than
either run alone supported.

This resolves the open thermal-timing caveat from session 4. There,
`ProcessInfo.thermalState` reported "serious" at 101s on battery vs. 347s
on power — a 3.4x difference that looked like it needed explaining. The
charts show the actual latency cliff landed in the same place (~102s)
both times. The variance was in the *reported* thermal state, not in the
hardware's actual behavior — `thermalState` is a poor predictor of actual
throttling and shouldn't be used to anticipate performance.

Secondary observations:

- The on-power run shows two plateaus: ~46ms for about four minutes, then
  a second step to ~49.4ms. The battery run skips the intermediate step
  and goes straight to ~49.5ms. Charging bought an intermediate step
  rather than a delay in reaching the same endpoint.
- The excursions around 270s and 520s (on-power run) appear in the scatter
  as dense bands of consecutive slow runs, not isolated outliers — i.e.
  sustained sub-plateaus, not noise.
