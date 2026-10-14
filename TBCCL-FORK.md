# About this fork

`tensorsofthewall/vllm-metal` is a maintained fork of [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) that carries one generic change: a pluggable pipeline-parallel transport seam (`vllm_metal/distributed/pipeline.py` and its wiring in the worker and model runner) that lets the Metal stage of a vLLM pipeline exchange activations through a transport chosen by the environment instead of the built-in one. [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) supplies that transport, which carries activations over [TBCCL](https://github.com/tensorsofthewall/tbccl) (TCP, including a direct Thunderbolt 4 link; not RDMA) between a Mac and a Linux CUDA host. The seam itself is generic and does not mention TBCCL. Everything else is upstream vllm-metal.

## What to install

Use the immutable tag, never `main`:

| Tag | Commit | Contents |
|---|---|---|
| `v0.31.0-tbccl1` | `fa4f934586bc27ac974a54183237594d1dc28fc5` | upstream vllm-metal v0.31.0 (`e7a0c90bec180e5dd4275c473fd7a36e709a930e`) plus the transport seam, one commit |

vllm-tbccl 0.2.0 is qualified against exactly this tag with vLLM 0.31.0. The tag is never moved. The current pairing is on the [TBCCL ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Provenance

- vllm-tbccl keeps a reference patch of the seam as provenance and checks automatically that the diff between the upstream base and this tag still equals it. Users never apply the patch; they install the tag.
- `main` of this fork follows upstream plus the seam. It is a development line, not a supported install target, and is updated by merging upstream and re-validating, never automatically.
- `tools/check_upstream_drift.py` and the weekly `upstream-drift` workflow report how far `main` is behind upstream and which upstream changes touch the files the seam changed (derived from the diff between the upstream base and the tag). They only report; they merge nothing and move no tag.
- To see exactly what the seam changes: `git diff e7a0c90bec180e5dd4275c473fd7a36e709a930e v0.31.0-tbccl1`.
