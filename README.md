# flicker

**FLUX.2 [klein] 4B inference, in JAX**

## model info

FLUX.2 [klein] (`black-forest-labs/FLUX.2-klein-4B`) is a step-distilled rectified-flow
transformer that denoises a 128-channel latent at 1/16 resolution with four Euler steps, no
classifier-free guidance and no KV cache. `flicker` reimplements each stage as a pure function over
an explicit parameter pytree — no Flax, no NNX:

```
prompt ──► Qwen3-4B encoder ──► ctx ──┐
                                      ▼
     initial latent ──► 4-step Euler (DiT) ──► latent ──► VAE decoder ──► image
```

The three stages load their weights one at a time, so the whole pipeline fits a single 16 GB
accelerator: the DiT (7.2 GiB) and the text encoder (8 GiB) do not co-reside.

## usage

```python
from flicker.config import KLEIN_4B
from flicker.pipeline import Paths, generate

paths = Paths(dit="flux-2-klein-4b.safetensors", text_index="text_encoder/", vae="vae/")

image = generate(paths, KLEIN_4B, "a misty forest at dawn", height=512, width=512, seed=0)
# image: (512, 512, 3) uint8
```

`generate` is host-side glue over three separable stages — `text_ctx`, `denoise_latent` and
`decode` — which is how the parity harness injects the reference's own conditioning and noise
instead of a tokenizer round trip.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
