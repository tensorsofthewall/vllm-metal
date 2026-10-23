# About this fork

`tensorsofthewall/vllm-metal` is a maintained fork of [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) that carries one generic change: a pluggable pipeline-parallel transport seam (`vllm_metal/distributed/pipeline.py` and its wiring in the worker and model runner) that lets the Metal stage of a vLLM pipeline exchange activations through a transport chosen by the environment instead of the built-in one. [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) supplies that transport, which carries activations over [TBCCL](https://github.com/tensorsofthewall/tbccl) (TCP, including a direct Thunderbolt 4 link; not RDMA) between a Mac and a Linux CUDA host. The seam itself is generic and does not mention TBCCL. Everything else is upstream vllm-metal.

## What to install

Use the immutable tag, never `main`:

| Tag | Commit | Contents |
|---|---|---|
| `v0.31.0-tbccl1` | `fa4f934586bc27ac974a54183237594d1dc28fc5` | upstream vllm-metal v0.31.0 (`e7a0c90bec180e5dd4275c473fd7a36e709a930e`) plus the transport seam, one commit |

vllm-tbccl 0.2.0 is qualified against exactly this tag with vLLM 0.31.0. The tag is never moved. The current pairing is on the [TBCCL ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Installing the TBCCL-compatible fork

Install uv 0.13.4 or later first (0.13.5 is current; `brew install uv`, `pipx install uv`, or the [uv installer](https://docs.astral.sh/uv/getting-started/installation/)). The `scripts/lib.sh` of the tag `v0.31.0-tbccl1` installs uv 0.9.18 when no uv is on the `PATH`, and that release trusts the emSign root that was revoked in October 2026 (see [Security](#security-emsign-ca-removal-october-2026)); with uv already installed the installer does not touch it.

```bash
git clone --branch v0.31.0-tbccl1 --depth 1 https://github.com/tensorsofthewall/vllm-metal.git
cd vllm-metal && ./install.sh
uv pip install --python .venv-vllm-metal/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0/vllm_tbccl-0.2.0-cp312-cp312-macosx_14_0_arm64.whl
.venv-vllm-metal/bin/python -m vllm_tbccl.info
```

`./install.sh` run **from the checkout, without arguments**, installs vLLM 0.31.0's prebuilt macOS wheel, installs this checkout editable and builds the Metal artifacts, all in `.venv-vllm-metal` inside the checkout; the vllm-tbccl wheel then goes into that same environment (so vLLM, vllm-metal and vllm-tbccl are in one environment). Python 3.12 is required (the installer selects it). The Homebrew formula, the `curl | bash` installer and `install.sh --stable`/`--dev` outside a checkout install upstream vllm-metal and lack the seam. The installer needs Apple silicon, Xcode's command-line tools and the Metal toolchain; no prebuilt wheel of the fork exists yet and none is claimed. vllm-tbccl's [installation guide](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) covers the CUDA node and the launch.

## Security: emSign CA removal (October 2026)

Private keys of eight eMudhra intermediate certificates were reported compromised in October 2026 and the affected emSign root (emSign ECC Root CA - G3) was removed from the CA bundles used by Python tooling (certifi 2026.10.19, pip 26.2.2, uv 0.13.4). What it means here:

- The fork pins no certifi, has no lockfile and ships no CA bundle: its Python dependencies are resolved when you install, so an installation made after 2026-10-20 gets certifi 2026.10.19 or later unless the environment already holds an older one.
- The only affected item is the uv bootstrap in `scripts/lib.sh`, which the immutable tag `v0.31.0-tbccl1` cannot change (uv 0.9.18, installed only when no uv is found). The tag stays as it is. On `main`, `scripts/lib.sh` installs uv 0.13.5. Install uv 0.13.4 or later yourself before running the installer from the tag; if the installer already put uv 0.9.18 on a machine, upgrade it.
- In any environment you create for this fork, make sure pip is 26.2.2 or later (`python -m pip install --upgrade pip`, or the hash-pinned wheel from the Python advisory when the installed pip is older) and certifi is 2026.10.19 or later.
- The documentation build (Read the Docs) installs pip 26.2.2 from its hash-pinned wheel and uv 0.13.5.

## Provenance

- vllm-tbccl keeps a reference patch of the seam as provenance and checks automatically that the diff between the upstream base and this tag still equals it. Users never apply the patch; they install the tag.
- `main` of this fork follows upstream plus the seam. It is a development line, not a supported install target, and is updated by merging upstream and re-validating, never automatically.
- `tools/check_upstream_drift.py` and the weekly `upstream-drift` workflow report how far `main` is behind upstream and which upstream changes touch the files the seam changed (derived from the diff between the upstream base and the tag). They only report; they merge nothing and move no tag.
- To see exactly what the seam changes: `git diff e7a0c90bec180e5dd4275c473fd7a36e709a930e v0.31.0-tbccl1`.
