# Writing

Scratch file for analogies and framings that fall out of the work. Raw material, nothing finished.

## Quantization as rounding a recipe book

A recipe says 240ml of milk, 5g of yeast, 3g of salt. Nobody measures 5.00000g of yeast — you round to what your scale can show, and the bread still comes out right. Quantization does the same thing to a model's weights: instead of storing each one as a 32-bit float with more precision than the result needs, you round it to fewer bits (int8, int4). Most of the "precision" in the original number was never load-bearing — the model degrades gracefully because, like the recipe, the outcome only cared about the measurement being close enough, not exact. The interesting question is the same in both cases: which measurements can you round aggressively before the loaf actually changes, and which ones (the leavening, maybe) are load-bearing enough that rounding wrecks it.

## The 40x lie in Core ML benchmarks

Most published Core ML benchmark numbers quietly report load time and inference time as if they're the same measurement. On the whisper-base encoder: compiling/loading the model takes ~1924ms, running it takes ~50ms. That's a 40x gap. A benchmark that times "predict()" after a fresh `MLModel(contentsOf:)` and reports the whole thing as "inference time" is describing a one-time cost as if it recurs on every call — which makes sense for a CLI tool that loads once and exits, and is wrong for literally any app that keeps the model warm. The fix isn't complicated, it's just rarely done: report load and inference as two separate numbers, because they answer two different questions (cold-start latency vs. steady-state throughput), and conflating them makes every number downstream of that benchmark off by an order of magnitude.

## What the sustained test is asking

The whole question is whether inference number 13,000 takes longer than inference number 5, because the phone got hot. Nobody publishes this because benchmarks measure sprints, and real usage is a marathon.

## An unverified number survives six days

I quoted the encoder's `.mlpackage` as "145MB on disk" from session 2 through session 4 — six days — before checking it against the actual file. It's 39MB; 145MB was an estimate for the full model, and we only ever converted the encoder. The memory-mapping finding (disk size costs single-digit MB in phys_footprint) still holds, just at the right disk size. Worth remembering: a number that sounds plausible and supports the point you're making is exactly the kind that doesn't get checked.

## Why Whisper

Speech recognition running locally with no network is the same problem a car has in a tunnel. Offline voice is exactly the constrained-device story I want to be known for — the benchmark and the positioning point the same direction.
