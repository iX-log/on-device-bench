# Parked
Things I want to add but am not adding before week 7.

- Second talk/article: using Xcode tooling to debug and inspect Core ML
  models on device. Instruments, `devicectl` container pulls, verifying
  which model actually loaded, reading `phys_footprint` from the Mach
  layer. Most of this method already exists as plumbing in this repo
  (device-pull scripts, `Memory.physFootprint()`, the picker-verification
  work from session 5), so this could be written up from existing notes
  rather than built as a new project.

- Run the Core ML Instruments template on all three precisions (fp16,
  int8, int4) to get per-layer timing and which compute unit each
  operation actually ran on. Could explain the session 5 finding: does
  the ANE expand int4 weights to a common working precision, or do some
  layers fall back to GPU? Would turn "int4 buys nothing, probably
  because compute-bound" into a claim with a layer trace behind it.
