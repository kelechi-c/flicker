# flicker

**FLUX.2 [klein] 4B text-to-image in raw JAX and Pallas: 306 ms per 1024² image on one TPU v6e chip.**

Each stage is a pure function over an explicit parameter pytree — no Flax, no NNX.

```
prompt ──► Qwen3-4B encoder (27 of 36 layers) ──► ctx ──┐
                                                        ▼
            noise ──► 4-step Euler (DiT) ──► latent ──► VAE decoder ──► image
```

## numbers

1024², batch 1, 4 steps, bf16, one TPU v6e-1 (Colab), p50 of 20 warm requests:

| engine | device | 1024² | 512² |
|---|---|---|---|
| **flicker** | TPU v6e-1 | **306 ms** (3.27 img/s) | **91.5 ms** |
| maxdiffusion (JAX) | TPU v6e-1 | 463.5 ms | 112.2 ms |
| sglang-diffusion | RTX Pro 6000 | 675.6 ms | — |
| official PyTorch | RTX Pro 6000 | 739.7 ms | — |
| plain JAX port (start) | TPU v6e-1 | 944.7 ms | — |

Stage split (text / DiT / VAE): 105 / 809 / 43 ms for the plain port, 7 / 270 / 26 ms now. MFU rose
from 17.6% to 54.5% of the v6e bf16 peak (918 TFLOP/s).

## what changed

- **text encoder (105 → 7 ms):** weights moved to the device once, and one `lax.scan` over 27 layers
  taps hidden states 9/18/27.
- **flash attention (DiT 784 → 388 ms):** 2304 tiles instead of naive SDPA.
- **fused qk-prep (`kernels.qk_prep`):** head transpose, QK-RMSNorm and RoPE in one Pallas pass,
  114 → 12 ms.
- **bounded attention (`kernels.attend`):** flash without the running max, legal because QK-norm
  bounds the logits (checked at load); 380 TFLOP/s against a 391 ceiling at head_dim 128.
- **dilated upsample (VAE 37 → 26 ms):** nearest-2x + 3x3 conv as one 4x4 conv.

`attention="flash"` and `upsample="nearest"` select the library kernel and the reference's literal
upsample (346 ms).

## quality

Gated against the PyTorch reference on 96 image pairs (32 prompts x 3 seeds), fed the reference's
own text conditioning: LPIPS median 0.0056 and step-latent SNR 36.4 dB pass their floors; DINOv2 p5
is 0.9819 against a 0.9835 floor, a miss inside the noise of a paired comparison. With flicker's
text encoder the images drift further from PyTorch (LPIPS ≈ 0.04), because bf16 padding-token states
are sensitive to operation order.

An int8 tier (258 ms, LPIPS 0.013) fails the gate and lives at the `int8-tier` tag, not on `main`.
Batching adds no throughput: at 4608 tokens the MXU is already saturated.

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

## tools

```bash
uv run python tools/generate.py "a misty forest at dawn" --size 1024 --grid 1   # images + prompts.json in out/images
uv run python tools/bench.py --reps 20 [--size 512] [--trace out/trace]        # p50 text / dit / vae / request, json on stdout
uv run python tools/op_profile.py out/trace --iterations 3                      # per-op time and TFLOP/s from an xprof trace
uv run python tools/xplane_to_perfetto.py out/trace                             # xplane -> perfetto json
uv run python tools/score.py REFERENCE_DIR CANDIDATE_DIR                        # lpips + dinov2 over same-named pngs
```

`score.py` needs torch, pyiqa and transformers; the rest need only flicker's dependencies.

## license

Apache-2.0 — see [`LICENSE`](LICENSE).
