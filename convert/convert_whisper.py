"""Convert the Whisper encoder to Core ML and verify numerics.

fp16 is a straight conversion via compute_precision. int8 and int4 convert
to fp16 first, then apply post-training weight compression from
coremltools.optimize.coreml (linear quantization for int8, palettization
for int4).
"""

import argparse

import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
    palettize_weights,
)
from transformers import WhisperForConditionalGeneration


class EncoderWrapper(torch.nn.Module):
    """torch.jit.trace needs a plain tensor back; HF returns an output object."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel).last_hidden_state


def default_out(model_name, precision):
    short = model_name.split("/")[-1]
    return f"models/{short}-encoder-{precision}.mlpackage"


def quantize(mlmodel, precision):
    if precision == "int8":
        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config = OptimizationConfig(global_config=op_config)
        return linear_quantize_weights(mlmodel, config=config)
    if precision == "int4":
        op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
        config = OptimizationConfig(global_config=op_config)
        return palettize_weights(mlmodel, config=config)
    raise ValueError(f"no quantizer for precision {precision!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--precision", choices=["fp16", "int8", "int4"], default="fp16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = default_out(args.model, args.precision)

    print(f"loading {args.model} ...")
    full = WhisperForConditionalGeneration.from_pretrained(args.model)
    full.eval()

    n_mels = full.config.num_mel_bins
    n_frames = full.config.max_source_positions * 2
    shape = (1, n_mels, n_frames)
    print(f"input shape: {shape}")

    wrapper = EncoderWrapper(full.model.encoder).eval()
    dummy = torch.randn(*shape)

    print("tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    print("converting ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="hidden_states")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    if args.precision != "fp16":
        print(f"quantizing to {args.precision} ...")
        mlmodel = quantize(mlmodel, args.precision)

    mlmodel.save(args.out)
    print(f"saved -> {args.out}")

    print("verifying ...")
    with torch.no_grad():
        ref = wrapper(dummy).numpy()

    got = mlmodel.predict({"mel": dummy.numpy()})["hidden_states"]

    diff = np.abs(ref - got)
    rel = diff.max() / (np.abs(ref).max() + 1e-9)

    tolerance = {"fp16": 0.02, "int8": 0.05, "int4": 0.15}[args.precision]

    print(f"  output shape     {got.shape}")
    print(f"  max abs diff     {diff.max():.6f}")
    print(f"  mean abs diff    {diff.mean():.6f}")
    print(f"  max rel diff     {rel:.6f}")
    print("  PASS" if rel < tolerance else f"  CHECK -- larger than expected for {args.precision}")


if __name__ == "__main__":
    main()
