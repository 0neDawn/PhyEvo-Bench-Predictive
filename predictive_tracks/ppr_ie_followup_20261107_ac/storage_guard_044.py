"""Collect unreachable hook cycles and release unused CUDA cache; no math changes."""
import gc
import time
from . import common as c


class StorageGuard:
    def __init__(self, adapter):
        import torch
        self.torch = torch
        self.native = adapter.model.forward
        self.calls = 0
        self.collections = 0
        self.collected_objects = 0
        self.released_allocated_bytes = 0
        self.released_reserved_bytes = 0
        self.maximum_boundary_allocated_bytes = 0
        self.maximum_boundary_reserved_bytes = 0
        assert torch.cuda.get_per_process_memory_fraction(0) == .45
        adapter.model.forward = self.forward

    def collect(self):
        torch = self.torch
        assert torch.cuda.get_per_process_memory_fraction(0) == .45
        before_a = torch.cuda.memory_allocated(0)
        before_r = torch.cuda.memory_reserved(0)
        count = gc.collect()
        torch.cuda.empty_cache()
        after_a = torch.cuda.memory_allocated(0)
        after_r = torch.cuda.memory_reserved(0)
        self.collections += 1
        self.collected_objects += count
        self.released_allocated_bytes += max(0, before_a-after_a)
        self.released_reserved_bytes += max(0, before_r-after_r)
        self.maximum_boundary_allocated_bytes = max(self.maximum_boundary_allocated_bytes, after_a)
        self.maximum_boundary_reserved_bytes = max(self.maximum_boundary_reserved_bytes, after_r)
        c.status(c.HERE/'monitor/storage_runtime_044.json', self.audit())

    def forward(self, *args, **kwargs):
        # The forecaster is still called exactly six times by the original loop.
        if self.calls % 6 == 0:
            self.collect()
        try:
            return self.native(*args, **kwargs)
        finally:
            self.calls += 1

    def audit(self):
        return dict(at=time.time(),forecaster_calls=self.calls,collections=self.collections,
            collected_unreachable_objects=self.collected_objects,
            released_allocated_bytes=self.released_allocated_bytes,
            released_reserved_bytes=self.released_reserved_bytes,
            maximum_boundary_allocated_bytes=self.maximum_boundary_allocated_bytes,
            maximum_boundary_reserved_bytes=self.maximum_boundary_reserved_bytes,
            per_process_fraction=.45,original_memory_limit_unchanged=True,
            no_input_output_weight_precision_or_forward_count_change=True,
            extra_GC_and_allocator_work_add_wall_time=True)


def install_factory():
    from predictive_tracks.ppr_dino_feedback_20261106_aa import worker
    build = worker.adapter_for
    def adapter_for(model):
        adapter = build(model)
        adapter._storage_guard_044 = StorageGuard(adapter)
        return adapter
    worker.adapter_for = adapter_for
