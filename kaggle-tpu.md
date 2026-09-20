# Kaggle TPU v5e-8 — command reference

Reference for driving Kaggle's free **TPU VM v5e-8** from the command line. Written
2026-09-18; verified against **Kaggle CLI 2.2.4** on rachel-arch (`kaggle quota` and
`kaggle competitions list` both returned live data for account `tensorkelechi`).

Anything here is safe to run from **this laptop** — the compute happens on Kaggle's
machines, so the "no GPU locally" rule does not apply to this path.

## 1. What the accelerator actually is

There is **no SSH, no IP, no remote endpoint**. Kaggle TPUs are attached to a *notebook
session* and exist only for that session's lifetime. "Connecting to the TPU" means getting
code to run inside a session that has one attached, in one of two ways:

| route | how | when |
| --- | --- | --- |
| interactive | browser notebook → *Settings* → *Accelerator* → **TPU VM v5e-8** | exploring, iterating on shapes |
| headless (`kaggle kernels push`) | local folder + `kernel-metadata.json` → runs as a batch version | repeatable runs, runs while you do something else |

`v5e-8` = **one host, 8 v5e chips**. So you should see **8 JAX devices**, and the mesh
runs over a single host's interconnect — no multi-host `torchrun`/jax-distributed patterns.

### Quota

`kaggle quota` is authoritative — it prints used / remaining / total and the weekly reset
time. As of 2026-09-18 17:5x: **TPU 0.00h used, 20.00h remaining, refresh 2026-09-19T00:00**;
GPU 30.00h total. Kaggle documents the TPU cap as **20 h/week** and **9 h maximum per
session**, and TPU allocations can sit in a queue.

```bash
kaggle quota
kaggle quota --format json
```

## 2. Install and auth

Installed with **uv tool** (not pip — the system Python is 3.14 and externally managed):

```bash
uv tool install kaggle        # ~/.local/bin/kaggle, isolated venv
uv tool upgrade kaggle        # update later
kaggle --version              # -> Kaggle CLI 2.2.4
```

Authentication, in preference order:

```bash
kaggle auth login             # OAuth loopback flow; opens a browser; caches ~/.kaggle/credentials.json
kaggle auth print-access-token            # mint a short-lived token (default 12h)
kaggle auth print-access-token --expiration 6h
```

Token alternatives, if you cannot use OAuth:

- `export KAGGLE_API_TOKEN=…`, or write the token to `~/.kaggle/access_token`
- legacy username/key pair in `kaggle.json` — note the CLI resolves that file to
  **`~/.config/kaggle/kaggle.json`** on Linux while `~/.kaggle/` does not exist (XDG), and
  `KAGGLE_CONFIG_DIR` overrides it

Verify auth before anything else:

```bash
kaggle competitions list      # any authenticated call works as a check
kaggle config view
```

Auth is **per machine**: victoria needs its own `kaggle auth login` (or a copy of
`~/.kaggle/credentials.json`) before it can push.

## 3. The metadata dir — `kaggle-tpu/`

```
hummingbird/kaggle-tpu/
  kernel-metadata.json    # what kaggle kernels push reads
  main.ipynb              # code_file — the notebook that runs on the v5e-8
```

```json
{
  "id": "tensorkelechi/hummingbird-tpu",
  "title": "hummingbird tpu",
  "code_file": "main.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "true",
  "enable_internet": "true",
  "machine_shape": "TpuV5E8",
  "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []
}
```

Fields that matter:

- **`machine_shape: "TpuV5E8"`** is the accelerator. Other shapes: `NvidiaTeslaT4`,
  `NvidiaL4`. `--accelerator <shape>` on `push` overrides it.
- **`enable_internet`** must be `"true"` for any `pip install` or download inside the kernel.
- **`id` must be `owner/slug`, and `title` must slugify to that slug** (`slugify("hummingbird tpu")
  == "hummingbird-tpu"`). If it does not, the CLI warns about "surprising behavior" and the
  rename can strand the kernel.
- `is_private` defaults to `true` when omitted.

### Only `code_file` is uploaded

Source-verified in `kaggle/api/kaggle_api_extended.py:kernels_push` — the request carries
`text = <contents of code_file>` and nothing else. **Other files sitting in `kaggle-tpu/`
do not reach the kernel.** To ship project code you must either

1. put it directly in the notebook, or
2. attach it with `dataset_sources` / `kernel_sources` (upload it as a private dataset or
   kernel first), or
3. fetch it inside the kernel (`enable_internet: "true"`) — e.g. `pip install git+https://…`
   or `git clone`.

`kaggle kernels init -p <dir>` writes a template metadata file; `kaggle kernels pull -k
<owner>/<slug> -p <dir> -m` downloads an existing kernel plus regenerated metadata.

## 4. Push / run / observe

```bash
cd hummingbird/kaggle-tpu

kaggle kernels push -p .                      # upload + start a new version
kaggle kernels push -p . --accelerator TpuV5E8
kaggle kernels push -p . -t 3600              # cap the run, seconds (server cap still applies)

kaggle kernels status   tensorkelechi/hummingbird-tpu      # latest run state
kaggle kernels logs     tensorkelechi/hummingbird-tpu      # logs from the latest run
kaggle kernels logs -f  tensorkelechi/hummingbird-tpu      # follow a running session, like tail -f
kaggle kernels files    tensorkelechi/hummingbird-tpu      # list output artefacts

kaggle kernels output   tensorkelechi/hummingbird-tpu -p out    # download outputs
kaggle kernels output   tensorkelechi/hummingbird-tpu --file-pattern '.*\.json' -p out
kaggle kernels pull     tensorkelechi/hummingbird-tpu -p . -m   # pull code + metadata back

kaggle kernels list -m --search hummingbird   # my kernels
kaggle kernels delete   tensorkelechi/hummingbird-tpu
```

`status` / `logs` / `output` all default to the **latest version**. Keep the slug in one
place in scripts rather than retyping it.

**A push is a batch run.** You cannot attach to an interactive session on version 2, and
you cannot edit a running version. For anything iterative, prototype in the browser
notebook, then push once the code is stable. Long pushes need no babysitting — poll
`status`, or `logs -f` if you want to watch.

Output files written under `/kaggle/working` come back with `kernels output`; write JSON
sidecars next to audio exactly as `scripts/generate.py` does locally, so `out/` stays
comparable across the GPU and TPU paths.

## 5. Inside the kernel

```python
import jax
print(jax.__version__, jax.device_count(), jax.devices())
# v5e-8 must report 8 devices, device_kind like "TPU v5 lite" / "TpuDevice"
```

`main.ipynb` in this folder runs exactly this, then a `jax.lax.psum` across the 8-device
mesh (sums `0..7` to `28.0`) and a single-device bf16 matmul throughput probe.

If JAX is not TPU-ready in the image:

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

PyTorch/XLA instead:

```bash
export PJRT_DEVICE=TPU
# matching pair, e.g. torch==2.8.0 + torch_xla[tpu]==2.8.0
python -c "import torch_xla; import torch_xla.runtime as xr; print(xr.device_count())"
```

Keras/TF: `tf.distribute.TPUStrategy`. **Version skew between JAX / `libtpu` /
`torch_xla` is the usual cause of "TPU not found"** — pin the trio together and pin a
Kaggle image version if a run must stay reproducible.

Timing traps that carry over from the GPU work: call `jax.block_until_ready()` before
trusting a duration, and remember the first call pays the compile — `docs/jax-tpu-port-decision.md`
puts one `jax.jit` around the whole denoising loop precisely so that cost is paid once.

## 6. Checking other things

```bash
kaggle quota                                    # weekly GPU/TPU budget, reset time
kaggle config set -n path -v ./out              # default download dir for a project
kaggle config view
kaggle competitions list -s arc
kaggle competitions submissions <competition>   # only if we enter one
kaggle datasets list -s 'stable audio'
kaggle datasets download <owner>/<dataset> -p out --unzip
kaggle datasets create -p <dir>                 # publishing a private dataset for kernel_sources
kaggle kernels list -m
kaggle kernels list -s 'stable audio' --sort-by voteCount   # or dateRun, hotness, relevance
```

Attach published artifacts as kernel inputs via the metadata lists:
`dataset_sources: ["owner/dataset"]`, `kernel_sources: ["owner/kernel"]`,
`model_sources: ["owner/model/framework/variation/version"]`.

## 7. Gotchas

- **No SSH into a Kaggle TPU**, and never will be — plan for push-and-poll or the browser.
- **Only `code_file` is uploaded** (see §3) — the most common surprise.
- `is_private` defaults to `true`; a pushed kernel is not automatically shared.
- `enable_internet` defaults to **`true`** in `kernels push` (and in the `kernels init`
  template) even though the published metadata docs say `false` — set it explicitly so the
  behaviour is not left to the default.
- TPU sessions are **single-host** and queue, so `status: queued` is normal.
- Exceeding the 9 h session cap kills the run; always pass `-t` for long jobs.
- Quota refreshes on Kaggle's schedule, not yours — check `kaggle quota` before a long sweep.
- The kernel's working dir is not this repo. Anything the notebook needs must be uploaded,
  attached as a source, or fetched at runtime.
