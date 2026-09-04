"""Convert the Whisper encoder to Core ML (fp16) and verify numerics."""

import argparse

import numpy as np
import torch
import coremltools as ct
from transformers import WhisperForConditionalGeneration


class EncoderWrapper(torch.nn.Module):
    """torch.jit.trace needs a plain tensor back; HF returns an output object."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel).last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--out", default="models/whisper-base-encoder-fp16.mlpackage")
    args = ap.parse_args()

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

    mlmodel.save(args.out)
    print(f"saved -> {args.out}")

    print("verifying ...")
    with torch.no_grad():
        ref = wrapper(dummy).numpy()

    got = mlmodel.predict({"mel": dummy.numpy()})["hidden_states"]

    diff = np.abs(ref - got)
    rel = diff.max() / (np.abs(ref).max() + 1e-9)

    print(f"  output shape     {got.shape}")
    print(f"  max abs diff     {diff.max():.6f}")
    print(f"  mean abs diff    {diff.mean():.6f}")
    print(f"  max rel diff     {rel:.6f}")
    print("  PASS" if rel < 0.02 else "  CHECK -- larger than expected for fp16")


if __name__ == "__main__":
    main()
