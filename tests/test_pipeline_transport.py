# SPDX-License-Identifier: Apache-2.0
"""Pluggable pipeline-parallel transport: default ring unchanged, external transport selected by qualname."""

import mlx.core as mx
import pytest

import vllm_metal.envs as envs
from vllm_metal.distributed import (
    MlxRingPipelineTransport,
    PipelineGroup,
    PipelineTransport,
)
from vllm_metal.distributed.pipeline import pipeline_recv, pipeline_send


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class RecordingTransport:
    """A fake external transport (referenced by qualname from the tests)."""

    instances: list = []

    def __init__(self, rank: int, size: int, peer_ips: list[str]) -> None:
        self.rank, self.size, self.peer_ips = rank, size, peer_ips
        self.sent: list = []
        self.closed = False
        RecordingTransport.instances.append(self)

    def send(self, x, dst):
        self.sent.append((tuple(x.shape), x.dtype, dst))
        return mx.array(0)

    def recv(self, shape, dtype, src):
        self.recv_args = (shape, dtype, src)
        return mx.ones(shape, dtype=dtype)

    def close(self):
        self.closed = True


class BrokenTransport:
    def __init__(self, rank, size, peer_ips):
        self.rank, self.size = rank, size


def _qual(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def test_default_group_wraps_ring_transport():
    pp = PipelineGroup(_FakeGroup(2, 1))
    assert isinstance(pp.transport, MlxRingPipelineTransport)
    assert (pp.rank, pp.size, pp.is_first, pp.is_last) == (1, 2, False, True)
    assert pp.group is pp.transport.group


def test_bootstrap_defaults_to_ring(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_PP_TRANSPORT_CLS", raising=False)
    called = {}

    def fake_ring(cls, rank, peer_ips):
        called["ring"] = (rank, peer_ips)
        return "ring"

    monkeypatch.setattr(PipelineGroup, "bootstrap_ring", classmethod(fake_ring))
    assert PipelineGroup.bootstrap(0, 2, ["a", "b"]) == "ring"
    assert called["ring"] == (0, ["a", "b"])


def test_bootstrap_selects_external_transport(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_PP_TRANSPORT_CLS", _qual(RecordingTransport))
    RecordingTransport.instances.clear()
    pp = PipelineGroup.bootstrap(1, 2, ["10.0.0.1", "10.0.0.2"])
    t = RecordingTransport.instances[0]
    assert pp.transport is t and pp.group is None
    assert (t.rank, t.size, t.peer_ips) == (1, 2, ["10.0.0.1", "10.0.0.2"])
    assert pp.is_last and not pp.is_first and isinstance(t, PipelineTransport)
    # recv/send go through the transport with the documented wire contract
    h = pipeline_recv(pp, 5, 8, mx.float32)
    assert h.shape == (1, 5, 8) and t.recv_args == ((5, 8), mx.float32, 0)
    pp0 = PipelineGroup(None, RecordingTransport(0, 2, ["a", "b"]))
    handle = pipeline_send(mx.zeros((1, 3, 8)), pp0)
    assert pp0.transport.sent == [((3, 8), mx.float32, 1)] and handle.item() == 0


def test_bad_transport_gives_clear_errors(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_PP_TRANSPORT_CLS", "no_such_module_xyz.Cls")
    with pytest.raises(ImportError, match="Cannot load pipeline transport"):
        PipelineGroup.bootstrap(0, 2, ["a", "b"])
    monkeypatch.setenv("VLLM_METAL_PP_TRANSPORT_CLS", "NotQualified")
    with pytest.raises(ValueError, match="package.module.Class"):
        PipelineGroup.bootstrap(0, 2, ["a", "b"])
    monkeypatch.setenv("VLLM_METAL_PP_TRANSPORT_CLS", _qual(BrokenTransport))
    with pytest.raises(TypeError, match="does not implement PipelineTransport"):
        PipelineGroup.bootstrap(0, 2, ["a", "b"])


def test_external_transport_does_not_need_peer_ips(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_PP_TRANSPORT_CLS", _qual(RecordingTransport))
    pp = PipelineGroup.bootstrap(0, 2)
    assert pp.size == 2 and pp.transport.peer_ips is None
    monkeypatch.delenv("VLLM_METAL_PP_TRANSPORT_CLS")
    with pytest.raises(ValueError, match="needs peer_ips"):
        PipelineGroup.bootstrap(0, 2)


def test_env_defaults():
    assert envs.VLLM_METAL_PP_TRANSPORT_CLS is None or isinstance(
        envs.VLLM_METAL_PP_TRANSPORT_CLS, str
    )
    assert envs.VLLM_METAL_DIST_BACKEND == "gloo"


def test_runner_passes_layout_to_optional_begin_step():
    seen = {}

    class T(RecordingTransport):
        def begin_step(self, packed, scheduler_output):
            seen["call"] = (packed, scheduler_output)

    t = T(0, 2, None)
    pp = PipelineGroup(None, t)
    # the runner's call shape (decode rows first, then prefill rows)
    begin = getattr(pp.transport, "begin_step", None)
    assert begin is not None
    begin(packed=[("d0", 1), ("p0", 5)], scheduler_output="sched")
    assert seen["call"] == ([("d0", 1), ("p0", 5)], "sched")
    assert (
        getattr(PipelineGroup(_FakeGroup(2)).transport, "begin_step", None) is None
    )  # ring: no hook
