# Parked
Parked experiments, not scheduled work.

- Run the Core ML Instruments template on all three precisions (fp16,
  int8, int4) to get per-layer timing and which compute unit each
  operation actually ran on. Could explain the session 5 finding: does
  the ANE expand int4 weights to a common working precision, or do some
  layers fall back to GPU? Would turn "int4 buys nothing, probably
  because compute-bound" into a claim with a layer trace behind it.

- Recovery experiment: run Sustained (10 min), stop, wait 5 minutes with
  the app closed, then run Quick. Does latency return to ~42ms, partially
  recover, or stay at ~49ms? Ten minutes of work, settles whether the
  README's "flat for the rest of the run" claim also holds after a
  cooldown, and answers a question any developer shipping a
  long-running inference feature will have.
