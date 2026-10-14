> **TBCCL fork.** This is a maintained fork of vllm-metal that carries one generic pluggable pipeline-parallel transport seam for [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl). Install the immutable tag `v0.31.0-tbccl1`, not `main`; see [TBCCL-FORK.md](TBCCL-FORK.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vllm-metal-dark.svg">
    <img src="docs/assets/vllm-metal-light.svg" alt="vLLM Metal — Run vLLM on your Apple Silicon GPU" width="900">
  </picture>
</p>

<p align="center">
| <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/"><b>Documentation</b></a> | <a href="https://docs.vllm.ai/projects/vllm-metal/en/latest/recipes/"><b>Recipes by Mac memory</b></a> | <a href="https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0"><b>Blog</b></a> | <a href="https://slack.vllm.ai/"><b>Developer Slack</b></a> (#hw-metal) |
</p>

vLLM Metal is a plugin that enables vLLM to run on Apple Silicon Macs using MLX as the primary compute backend. It unifies MLX and PyTorch under a single lowering path.

---
*Latest News* 🔥

- [2026/08] vLLM Metal now uses M5 NAX tensor units to accelerate MHA, GQA, and MQA prefill.
- [2026/08] Qwen3.8 now runs on Metal! `mlx-community/Qwen3.8-27B-8bit` serves a 27B hybrid SDPA + GDN linear model on a single Apple Silicon Mac.
- [2026/04] We released the new version v0.2.0! Unified paged varlen Metal kernel is now the default attention backend. 83x TTFT, 3.6x throughput compared to v0.1.0.

---

## Installing the TBCCL-compatible fork

This repository is a maintained fork of upstream [vllm-metal](https://github.com/vllm-project/vllm-metal). It is required only to run a vLLM pipeline whose Metal stage talks to a Linux CUDA node through [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) over [TBCCL](https://github.com/tensorsofthewall/tbccl): it adds one generic pluggable pipeline-parallel transport seam and nothing TBCCL-specific. For everything else use upstream vllm-metal.

**Supported integration: the immutable tag `v0.31.0-tbccl1`** (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`, upstream vllm-metal 0.31.0 plus the seam), qualified with vLLM 0.31.0 and vllm-tbccl 0.2.0. Never install `main` for TBCCL use. Install it from a checkout of the tag with the installer **in that checkout**, then add the matching vllm-tbccl wheel to the same environment (the installer creates the Python 3.12 environment `.venv-vllm-metal` inside the checkout; it needs Apple silicon, Xcode's command-line tools and the Metal toolchain):

```bash
git clone --branch v0.31.0-tbccl1 --depth 1 https://github.com/tensorsofthewall/vllm-metal.git
cd vllm-metal && ./install.sh
uv pip install --python .venv-vllm-metal/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0/vllm_tbccl-0.2.0-cp312-cp312-macosx_14_0_arm64.whl
.venv-vllm-metal/bin/python -m vllm_tbccl.info      # reports vllm-tbccl 0.2.0, C ABI 1, wire protocol 4
```

Two ways of installing vllm-metal do **not** give you the fork, and must not be used for this pairing: the **Homebrew formula** and the **`curl ... | bash` installer** in the upstream installation instructions below install upstream's own releases (no transport seam), and `./install.sh --stable` or `--dev` run outside a checkout does the same. The fork's GitHub Releases page also lists automatically built development prereleases of `main`; they are not the supported tag. A prebuilt wheel of the fork does not exist yet, so the tag is built from source on the Mac. The activation, a launch command and the CUDA node are described in [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl#installation). More about the fork, its provenance and the tag: [TBCCL-FORK.md](TBCCL-FORK.md).

## Architecture

Upstream vLLM supplies the API server, scheduler, and paged block manager; mlx_lm supplies the token-wise model layers; vllm-metal owns the request-aware attention path — the paged varlen kernel, M5 NAX prefill, and speculative decoding.

![vllm-metal in the vLLM stack](https://raw.githubusercontent.com/vllm-project/vllm-metal/main/docs/assets/architecture.svg)

## Requirements

- macOS 15 (Sequoia) or later, on Apple Silicon

## Supported Models

vllm-metal supports a growing set of models on Apple Silicon. See the full matrix in [docs/supported_models.md](docs/supported_models.md).

## Installation

These are the **upstream** installation methods. They install upstream vllm-metal, not the TBCCL-compatible fork; for the fork see [Installing the TBCCL-compatible fork](#installing-the-tbccl-compatible-fork).

- **Stable release:** Install with Homebrew, then run `vllm` without activating an environment.

  ```bash
  brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
  brew install vllm-project/vllm-metal/vllm-metal
  ```

- **Latest development build:** Use the curl installer, then activate its environment. No compiler needed.

  ```bash
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
  source ~/.venv-vllm-metal/bin/activate
  ```

- **Contributing Python or kernel code:** Follow the [source setup](docs/CONTRIBUTING.md#development-setup) for an editable checkout and native build tools.
