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

## What this means for your app

- **int8, not int4.** Half the size for 0.4 points of WER. int4 halves it again and triples the errors.
- **Load at launch.** First load after install is ~2s; every launch after is ~135ms.
- **Ignore file size.** A 39MB model costs 6–9MB of RAM: Core ML mmaps weights.
- **Budget with os_proc_available_memory().** A 6GB device gives you ~3GB.
- **You get 100 seconds.** Then 18% slower and flat for the rest of the run.
- **Don't trust thermalState.** It fired at 101s and 347s while the actual slowdown was at 102s both times.
- **Unplug before you measure.** Charging reads 20% fast. Low Power Mode costs 56%.

Measured on one model (Whisper-base encoder, ~20M params) on one device. The direction of each finding should generalise, the magnitudes may not.

## Key Findings

| Measurement | Result | Session |
|---|---|---|
| Steady-state inference | 43.0ms median, 44.2ms p95 (100 runs) | 6 |
| Load time, cold vs. warm | 1975ms → 135ms | 2 |
| Sustained-run behavior | Flat ~41.7ms until ~102s, then a near-vertical step to a ~49ms plateau (a cliff, not a slope) | 8 |
| Quantization vs. speed | int4 is 4x smaller than fp16 but only ~3% faster (compute-bound, not memory-bound) | 5 |
| Quantization vs. accuracy | WER 3.4% (fp16) → 3.8% (int8) → 8.8% (int4) | 7 |
| Memory ceiling before jetsam kill | ~3060MB on a 6GB device (half of RAM, not most of it) | 9 |

Conditions, caveats, and raw data: [RESULTS.md](RESULTS.md).

## Charts

![Overlay of two sustained runs showing latency flat at ~41.7ms then stepping near-vertically to a plateau at ~102 seconds on both battery and power](results/charts/sustained_comparison.png)

![Bar chart of disk size against a line of aggregate WER across fp16, int8, and int4, showing int8 barely raises error while int4 nearly triples it for a further 2x size reduction](results/charts/quantization-tradeoff.png)

## Limitations

- Core ML only, encoder only: no comparison to ONNX Runtime, TFLite, or whisper.cpp, and no end-to-end on-device transcription number (the decoder ran on a Mac CPU throughout).
- n=1 device, one ~20M-parameter model: unit-to-unit variance, other model architectures, and larger-model behavior are all unmeasured.
- Most figures are a single session's output, not repeat-checked for run-to-run variance.

Everything else, including per-session caveats and what's still
unexplained, is in [RESULTS.md](RESULTS.md).

## Reproducing

Setup and full step-by-step instructions: [REPRODUCING.md](REPRODUCING.md).

## License

Code is [MIT](LICENSE). Measurement data under `results/` (raw JSON,
charts, screenshots) is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/):
cite this repo if you use the numbers.

## Author

Ixhen Hasani, [ix-dev.com](https://ix-dev.com)

Cross-device data is this project's biggest gap: everything here is one
iPhone 14 Pro Max (A16, 6GB). If a number here doesn't reproduce on your
own run, or you run this on other hardware (an A17 or A18 result would
be genuinely useful), please open an issue with your conditions and
output.
