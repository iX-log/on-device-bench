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

- **int8, not int4.** Half the size for 0.4 points of WER. int4 halves it again, costs 2.6x the errors, and costs *more* memory at load, not less.
- **Load at launch.** First load after install is ~2s; every launch after is ~135ms.
- **File size is not memory size, in either direction.** A 39MB fp16 model costs 6–9MB of RAM on warm launches, because Core ML mmaps weights. But on a cold load the 10MB int4 model cost 66MB of footprint, more than fp16 did: palettized weights are expanded at load. Measure footprint, never infer it from the file.
- **Budget with os_proc_available_memory(), and don't assume a bigger phone helps.** The call returns a fixed per-device limit minus your current footprint, arithmetically, so you can budget precisely. But the limit does not scale with RAM: an 8GB iPhone 16 Pro and a 12GB iPhone 17 Pro Max return the identical 3,539,992,576 bytes. There is no phone you can buy your way past.
- **Sustained degradation is device-specific, in shape and size.** Four devices, three shapes: the A16 holds flat then steps, the A19 and A19 Pro creep smoothly with no step, the A18 Pro does both in sequence. Final drift ranges from +7% to +28% and does not improve with chip generation. Measure the device you ship to.
- **Cooldown matters more than silicon.** +12.2% drift cooled vs. +46.1% warm on the same device: a bigger swing than two chip generations.
- **thermalState reports late.** On the A18 Pro the slowdown was half-applied by 85.5s and `serious` was not reported until 90.6s, while `fair` arrived 37s before any latency change at all. It confirms throttling that has already happened; it does not warn you.
- **Unplug before you measure, and treat Low Power Mode as a different distribution.** Charging made the same benchmark read about 20% slower. Low Power Mode cost +94% on the A19 Pro against +56% on the A16, and widened the latency spread from 1.3ms to 21.4ms. Under it a 2026 phone is slower than a 2022 phone at full speed.

Measured on one model (Whisper-base encoder, ~20M params) across four devices (A16, A18 Pro, A19, A19 Pro). Of nine findings published here before cross-device testing, six turned out to be properties of one phone rather than of iOS. Treat every magnitude below as a starting point to re-measure, not a constant.

## Key Findings

| Measurement | Result | Session |
|---|---|---|
| Steady-state inference | 43.0ms median, 44.2ms p95 (100 runs) | 6 |
| Load time, cold vs. warm | 1975ms → 135ms | 2 |
| Sustained-run behavior | Device-specific in shape: A16 steps, A19 and A19 Pro creep, A18 Pro steps then creeps. Final drift +7% to +28%, not monotonic with generation | 8, 10, 11, 12 |
| Quantization vs. speed | int4 is 4x smaller than fp16 but only ~3% faster (likely compute-bound, not confirmed with a layer trace) | 5 |
| Quantization vs. accuracy | WER 3.4% (fp16) → 3.8% (int8) → 8.8% (int4) | 7 |
| Memory ceiling before jetsam kill | Fixed per device and does not scale with RAM: 3072 MiB on the A16 (54% of its reported 5.505 GiB), 3376 MiB on both the 8GB A18 Pro and the 12GB A19 Pro | 9, 11, 12 |
| Encoder output determinism | Bit-identical across chip generations and quantization schemes; differs across iOS versions by mean 0.0025, which does not change WER | 12 |
| Low Power Mode | +56% on the A16, +94% on the A19 Pro, and 16x wider latency spread | 3, 12 |

Conditions, caveats, and raw data: [RESULTS.md](RESULTS.md).

## Charts

![Overlay of two sustained runs showing latency flat at ~41.7ms then stepping near-vertically to a plateau at ~102 seconds on both battery and power](results/charts/sustained_comparison.png)

![Seven sustained runs normalised to each run's own first-minute median, plotted as percent slower over 600 seconds. Three iPhone 14 Pro Max (A16) runs hold flat for about 105 seconds, then two step to roughly 11 percent before stepping again to about 19 percent, while the third steps straight to 18 percent. The iPhone 16 Pro (A18 Pro) steps to about 10 percent at 85 seconds, plateaus, then climbs steadily to +28.1 percent. The iPhone 17 (A19) cooled run creeps to about 17 percent and falls back, ending at +12.1 percent. The iPhone 17 Pro Max (A19 Pro) is the flattest line, a gentle creep to +6.8 percent. The A19 warm-start run climbs steeply and noisily above all others, ending at +46.1 percent.](results/charts/cross-device-normalised.png)

![Bar chart of disk size against a line of aggregate WER across fp16, int8, and int4, showing int8 barely raises error while int4 raises it 2.6x for a further 2x size reduction](results/charts/quantization-tradeoff.png)

## Limitations

- Core ML only, encoder only: no comparison to ONNX Runtime, TFLite, or whisper.cpp, and no end-to-end on-device transcription number (the decoder ran on a Mac CPU throughout).
- n=4 devices (A16, A18 Pro, A19, A19 Pro), all iPhones, one ~20M-parameter model: unit-to-unit variance within a model, iPads, other architectures, and larger-model behavior are all unmeasured. Three of the four were borrowed for a single afternoon and cannot be re-run.
- Most figures are a single session's output, not repeat-checked for run-to-run variance. The sustained runs on the A18 Pro and A19 Pro are n=1 each.
- Raw per-run data is committed for the seven sustained runs (sessions 3, 4, 10, 11, 12) and for four memory-ceiling probes (sessions 9, 11, 12). Quick-test figures write no file, so sessions 1, 2, 5, 6 and the precision and Low Power Mode tables in sessions 11 and 12 are evidenced by screenshots in `results/screenshots/` rather than by a distribution you can re-derive.
- Encoder feature dumps from the borrowed devices are held out of git for size. The comparison results in session 12 are reproducible from the committed September 11 dumps plus a re-run on your own hardware, not from files in this repo.
- The finding that iOS version changes encoder output is a natural experiment, not a controlled one: four dump sets split cleanly along OS version with no other variable separating them, but nobody changed the OS on a single handset and re-measured.

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

This repo now covers four devices: an iPhone 14 Pro Max (A16), an
iPhone 16 Pro (A18 Pro), an iPhone 17 (A19) contributed by a tester, and
an iPhone 17 Pro Max (A19 Pro). Three of them were borrowed for an
afternoon, and that afternoon overturned six of the nine findings
published here beforehand.

The remaining gaps are an A17, anything below an A16, and any iPad. Low
Power Mode has been measured on two of the four devices, and the
cold-load footprint figures on one. If a number here doesn't reproduce
on your own run, or you run this on other hardware, please open an issue
with your conditions and output.
