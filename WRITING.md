# Writing

Scratch file for analogies and framings that fall out of the work. Raw material, nothing finished.

## Quantization as rounding a recipe book

A recipe says 240ml of milk, 5g of yeast, 3g of salt. Nobody measures 5.00000g of yeast; you round to what your scale can show, and the bread still comes out right. Quantization does the same thing to a model's weights: instead of storing each one as a 32-bit float with more precision than the result needs, you round it to fewer bits (int8, int4). Most of the "precision" in the original number was never load-bearing: the model degrades gracefully because, like the recipe, the outcome only cared about the measurement being close enough, not exact. The interesting question is the same in both cases: which measurements can you round aggressively before the loaf actually changes, and which ones (the leavening, maybe) are load-bearing enough that rounding wrecks it.

## The 40x lie in Core ML benchmarks

Most published Core ML benchmark numbers quietly report load time and inference time as if they're the same measurement. On the whisper-base encoder: compiling/loading the model takes ~1924ms, running it takes ~50ms. That's a 40x gap. A benchmark that times "predict()" after a fresh `MLModel(contentsOf:)` and reports the whole thing as "inference time" is describing a one-time cost as if it recurs on every call, which makes sense for a CLI tool that loads once and exits, and is wrong for literally any app that keeps the model warm. The fix isn't complicated, it's just rarely done: report load and inference as two separate numbers, because they answer two different questions (cold-start latency vs. steady-state throughput), and conflating them makes every number downstream of that benchmark off by an order of magnitude.

## What the sustained test is asking

The whole question is whether inference number 13,000 takes longer than inference number 5, because the phone got hot. Nobody publishes this because benchmarks measure sprints, and real usage is a marathon.

## An unverified number survives six days

I quoted the encoder's `.mlpackage` as "145MB on disk" from session 2 through session 4 (six days) before checking it against the actual file. It's 39MB; 145MB was an estimate for the full model, and we only ever converted the encoder. The memory-mapping finding (disk size costs single-digit MB in phys_footprint) still holds, just at the right disk size. Worth remembering: a number that sounds plausible and supports the point you're making is exactly the kind that doesn't get checked.

## One number can't describe a sustained run

On the 10-minute battery run, mean was 47.76ms but median was 49.2ms: the mean sits below the median, which is backwards for latency. Latency is normally right-skewed: a hard floor, plus occasional slow outliers pulling the mean up. Here the skew points left because this isn't one distribution. It's the first minute at ~41.6ms and the last minute at ~49.4ms glued together. Most of the ten minutes was spent throttled, so the median lands in the throttled regime while the fast early runs form a long left tail.

The stdev of 3.2ms sounds tight but it's describing two different machines averaged together. The point: on a sustained run the population changes underneath you, so no single summary statistic is meaningful. Report first minute and last minute separately. For contrast, the same model over 100 runs in four seconds had a spread under 1ms, because the phone never left one thermal state. Same model, same device, completely different distributions depending only on how long you ran it.

## The summary said slope, the chart said cliff

The sustained-run summary reported first minute 41.8ms, last minute 49.4ms, drift +18%. That reads as gradual degradation. Plotting the same 12,848 samples showed something else entirely: flat until 102 seconds, then a near-vertical step, then a plateau. Same data, opposite mental model, and completely different advice for anyone building an app: "expect a slow creep" versus "expect a cliff at ~100s, budget for the state change, not the average."

Publish raw data alongside summary numbers. Any statistic that collapses a time series into one figure can hide the shape that actually matters. Pairs with "One number can't describe a sustained run" above: both are cases where the honest summary was still the wrong summary.

## Why Whisper

Speech recognition running locally with no network is the same problem a car has in a tunnel. Offline voice is exactly the constrained-device story I want to be known for: the benchmark and the positioning point the same direction.

## The practical payload

The seven what-this-means-for-your-app lines (the takeaway list for both the article and the talk's closing slide):

- Use int8, not int4. Half the download for 0.4 points of WER. int4 halves it again but nearly triples errors and buys almost no speed.
- Load the model at launch, not on first use. First load after install is ~2s of on-device compilation; every launch after is ~135ms. Lazy loading on first tap is a two-second stall that looks like a bug.
- Don't size models against file size. A 39MB .mlpackage costs 6–9MB of footprint. Core ML memory-maps weights.
- Budget against os_proc_available_memory(), not device RAM. A 6GB device gives an app ~3GB, and that API counted down accurately to the kill.
- Design for 100 seconds, not 10 minutes. Full speed for ~100s, then ~18% slower and flat. Pace long-running features or set expectations.
- Don't branch on ProcessInfo.thermalState. It reported "serious" at 101s in one run and 347s in another while the actual slowdown happened at the same point both times.
- Benchmark unplugged. Charging reads ~20% fast; Low Power Mode costs ~56%.

Measured on one model (Whisper-base encoder, ~20M params) on one device. The direction of each finding should generalise, the magnitudes may not.

## Opening lines: four seconds to land it

A developer scrolling gives a title and one line before deciding whether to keep reading. No preamble, no explaining what Whisper is; that has to come later or not at all.

Three candidate openings:

1. "A 39MB model costs 8MB of RAM."
2. "You get 100 seconds, then you're on a different machine."
3. "4x smaller, 3% faster, 2.6x less accurate."

The first is strongest. A reader can check it against something they've already done (a bundled asset, a memory graph in Instruments) without first needing to know anything about Whisper, quantization, or thermal states. The other two require the cliff or the quantization framing to already be loaded into the reader's head before the number means anything.

The rule: the number is line one, the explanation is line two, and "how I found out" comes no earlier than paragraph three. Same rule for titles: "What I learned benchmarking Core ML" gets skipped; "A 39MB model costs 8MB of RAM" gets clicked.
