# flicker

**FLUX.2 [klein] 4B text-to-image inference in raw JAX, tuned for one TPU.**

Each stage is a pure function over an explicit parameter pytree — no Flax, no NNX.

```
prompt ──► Qwen3-4B encoder (27 of 36 layers) ──► ctx ──┐
                                                        ▼
            noise ──► 4-step Euler (DiT) ──► latent ──► VAE decoder ──► image
```

## numbers

1024², batch 1, 4 steps, bf16, one TPU v6e-1 (Colab), p50 of 20 warm requests:

| | request | text / DiT / VAE |
|---|---|---|
| default | 346 ms (2.89 img/s) | 7 / 298 / 37 ms |
| `attention="bounded"` + `upsample="dilated"` | 306 ms (3.26 img/s) | 7 / 270 / 26 ms |
| `gemm="smooth_large"` (int8, lossy) | 300 ms | 7 / 251 / 37 ms |

The last two rows are opt-in. `attention="bounded"` is an attention kernel without flash's running
max (legal because QK-norm bounds the logits); `upsample="dilated"` runs the VAE's nearest-2x + 3x3
conv as one 4x4 conv. Together they pass two of the three quality conditions and miss the third
narrowly, so they are off by default.

The DiT and VAE are gated against the PyTorch reference on 96 image pairs, fed the reference's
own text conditioning. With flicker's text encoder the images drift further from PyTorch
(LPIPS ≈ 0.04), because bf16 padding-token states are sensitive to operation order.

## usage

Weights are the files of `black-forest-labs/FLUX.2-klein-4B`.

```python
from flicker.config import Config
from flicker.pipeline import Paths, generate_resident, load_all

cfg = Config()
paths = Paths(
    dit="flux-2-klein-4b.safetensors",
    text_index="text_encoder/model.safetensors.index.json",
    vae="vae/diffusion_pytorch_model.safetensors",
)

weights = load_all(paths, cfg)  # ~13 GB on device: needs a 32 GB accelerator
image = generate_resident(weights, cfg, "a misty forest at dawn", height=1024, width=1024)
# image: (1024, 1024, 3) uint8
```

On a 16 GB accelerator use `generate(paths, cfg, prompt)`, which loads one stage at a time.

The Pallas kernels (flash attention, fused q/k prep) run on TPU only; every other backend takes
the plain XLA path.

## license

Apache-2.0 — see [`LICENSE`](LICENSE).
