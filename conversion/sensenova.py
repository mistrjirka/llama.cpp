"""SenseNova-Loop: a Qwen3.5 hybrid trunk with a repeated block and per-sublayer residual routers.

Two facts drive this converter.

The loop (`loop_settings`) runs the physical layers `start..end-1` `count` times with shared weights.
llama.cpp has no notion of a layer execution plan, so the loop is expanded here: every visit becomes
its own block in the file, with its own recurrent state and KV slots. The arithmetic is identical,
because each visit is the same weights applied to a fresh state. Cost is file size, not correctness:
24 of 32 layers are stored twice.

The mHC routers (`mhc`) are per-sublayer, not per-layer: `mhc_attn` and `mhc_ffn` each own an fp32
projection of width 2*streams*dim -> 2*streams + streams^2, and the sublayer runs once on the
pre-weighted sum of the residual streams. Those tensors go into the file untouched and stay fp32.

Vision and the MTP head are dropped: this GGUF is text-only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
import gguf
from . import ModelBase, logger
from .qwen import Qwen3_5TextModel


def execution_plan(loop: dict, n_phys: int) -> list[int]:
    """Physical layer order for one token, exactly as their `execution_plan` builds it."""
    if not isinstance(loop, dict):
        raise ValueError("loop_settings must be a mapping")
    start, end = int(loop["start"]), int(loop["end"])
    counts = loop.get("counts")
    if not isinstance(counts, list) or len(counts) != 1:
        raise ValueError(f"only a single loop count is supported, got {counts}")
    count = int(counts[0])
    if int(loop.get("eval_count", count)) != count:
        raise ValueError("eval_count and counts disagree")
    if not 0 <= start < end <= n_phys:
        raise ValueError(f"loop range {start}..{end} does not fit {n_phys} layers")
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n_phys))


def mhc_kinds(plan: list[int], mhc: dict | None) -> list[int]:
    """Per virtual block: 0 none, 1 enters the streams, 2 leaves them, 3 both, 4 plain router."""
    kinds = [0] * len(plan)
    if not mhc:
        return kinds
    lo, hi = int(mhc["start"]), int(mhc["end"])
    inside = lambda p: p is not None and lo <= p < hi
    for v, p in enumerate(plan):
        if not inside(p):
            continue
        prev_p = plan[v - 1] if v > 0 else None
        next_p = plan[v + 1] if v + 1 < len(plan) else None
        enters = not inside(prev_p) or p <= prev_p          # first visit of a pass
        leaves = not inside(next_p) or next_p <= p          # last visit of a pass
        kinds[v] = 3 if (enters and leaves) else 1 if enters else 2 if leaves else 4
    return kinds


@ModelBase.register("SenseNovaLoopForConditionalGeneration", "SenseNovaLoopModel")
class SenseNovaLoopModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.SENSENOVA_LOOP
    no_mtp = False

    def __init__(self, dir_model, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)

        loop = self.hparams.get("loop_settings")
        mhc = self.hparams.get("mhc")
        n_phys = int(self.hparams["num_hidden_layers"])

        if mhc and float(mhc.get("pre_scale", 1.0)) != 1.0:
            raise ValueError("mHC pre_scale is not 1.0; the graph would need it baked in")
        if mhc and mhc.get("fixed_identity"):
            raise ValueError("mHC fixed_identity is set; the mixing matrix would be the identity")

        self.plan = execution_plan(loop, n_phys)
        self.mhc = mhc
        self.kinds = mhc_kinds(self.plan, mhc)
        self.n_virtual = len(self.plan)

        self.virtual_for: dict[int, list[int]] = {}
        for v, p in enumerate(self.plan):
            self.virtual_for.setdefault(p, []).append(v)

        # the file describes the expanded plan, so the block count and name map follow it
        self.n_phys = n_phys
        self.n_mtp = 0 if self.no_mtp else int(self.hparams.get("mtp_num_hidden_layers", 0))
        self.block_count = self.n_virtual + self.n_mtp
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        logger.info(f"loop plan: {n_phys} physical layers -> {self.n_virtual} blocks, "
                    f"mHC on {sum(1 for k in self.kinds if k)} of them")

    def set_gguf_parameters(self):
        # the base writes the recurrent-layer array from the physical layer types; this file
        # describes the expanded plan, so suppress it and write the array for the plan instead
        layer_types = self.hparams.pop("layer_types", None)
        n_hidden = self.hparams["num_hidden_layers"]
        self.hparams["num_hidden_layers"] = self.n_virtual  # mixin writes nextn = block_count - this
        try:
            super().set_gguf_parameters()
        finally:
            self.hparams["num_hidden_layers"] = n_hidden
            if layer_types is not None:
                self.hparams["layer_types"] = layer_types

        if layer_types is not None:
            self.gguf_writer.add_recurrent_layers([layer_types[p] == "linear_attention" for p in self.plan] + [False] * self.n_mtp)

        if self.mhc:
            arch = self.gguf_writer.arch
            streams = int(self.mhc["streams"])
            self.gguf_writer.add_uint32(gguf.Keys.MHC.STREAMS.format(arch=arch), streams)
            self.gguf_writer.add_array(gguf.Keys.MHC.LAYER_KINDS.format(arch=arch), self.kinds + [0] * self.n_mtp)
            logger.info(f"mHC: {streams} streams, {int(self.mhc['iterations'])} sinkhorn iterations")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.visual."):
            return  # text-only: vision tower stays out of this file

        m = re.match(r"model\.(?:language_model\.)?layers\.(\d+)\.(.+)$", name)
        if m:
            phys, tail = int(m.group(1)), m.group(2)
            if phys >= self.n_phys:
                # MTP block (filter_tensors put it at n_phys + i): it follows the expanded plan
                v = self.n_virtual + (phys - self.n_phys)
                yield from super().modify_tensors(data_torch, f"model.layers.{v}.{tail}", v)
                return
            if tail.endswith("conv1d.weight"):
                # ssm_conv reads an F32 kernel, so this one is never narrowed
                data_torch = data_torch.to(torch.float32)
            # the loop is executed, not stored: only the first visit owns bytes in the file, the
            # loader aliases the later visits onto these tensors
            for v in self.virtual_for[phys][:1]:
                yield from super().modify_tensors(data_torch, f"model.layers.{v}.{tail}", v)
            return

        yield from super().modify_tensors(data_torch, name.replace("model.language_model.", "model."), bid)
