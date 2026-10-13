# SPDX-License-Identifier: Apache-2.0
"""Distributed primitives for vLLM Metal (pipeline parallelism)."""

from vllm_metal.distributed.pipeline import (
    MlxRingPipelineTransport,
    PipelinedModel,
    PipelineGroup,
    PipelineTransport,
    apply_pipeline_split,
    is_non_last_stage,
    pipeline_send,
)

__all__ = [
    "MlxRingPipelineTransport",
    "PipelineTransport",
    "PipelineGroup",
    "PipelinedModel",
    "apply_pipeline_split",
    "is_non_last_stage",
    "pipeline_send",
]
