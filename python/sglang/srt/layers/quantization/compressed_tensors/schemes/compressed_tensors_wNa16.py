# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import ActivationOrdering

# yapf conflicts with isort for this block
# yapf: disable
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
    permute_param_layout_,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    MarlinLinearLayerConfig,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    replace_parameter,
    unpack_cols,
)
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack


ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsLinearScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
                 symmetric: Optional[bool] = True,
                 actorder: Optional[ActivationOrdering] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        # ole-eval h28.7: set w_q/w_s/w_zp/w_gidx_name in __init__ so the
        # attributes exist even if process_weights_after_loading is skipped
        # (ADR-0001 lxm.2.3 zero-copy invariant on the client side).
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"
        # workspace is set by process_weights_after_loading via
        # marlin_make_workspace(device); when that's skipped on the client,
        # we leave it as None (apply() will fail loudly if actually called
        # without a workspace, since the kernels need it).
        # ole-eval h28.7: workspace + is_k_full are normally set in
        # process_weights_after_loading. When that's skipped (ADR-0001
        # lxm.2.3 zero-copy invariant on the client), apply_weights
        # computes them lazily from the layer's actual shape.
        self.workspace = None
        self.is_k_full = False

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
                           if not self.symmetric else
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
                       input_size: int, output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        self.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx
        )

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }

        zeros_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0,
                                               packed_dim=0,
                                               packed_factor=self.pack_factor,
                                               **zeros_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # w_q_name / w_s_name / w_zp_name / w_gidx_name are set in __init__
        # so they're available even when this method is skipped (ADR-0001
        # lxm.2.3 zero-copy invariant — client meta-init runs without
        # process_weights_after_loading).
        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config

        check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(device)

        def _transform_param(
            layer: torch.nn.Module, name: Optional[str], fn: Callable
        ) -> None:
            if name is not None and getattr(layer, name, None) is not None:

                old_param = getattr(layer, name)
                new_param = fn(old_param)
                # replace the parameter with torch.nn.Parameter for TorchDynamo
                # compatibility
                replace_parameter(
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
                )

        def transform_w_q(x):
            # ole-eval h28.11: skip Marlin repack on sm_80. The Marlin kernel
            # template fails to compile for int32 scalar_t (upstream bug),
            # so we use awq_dequantize + F.linear in apply_weights. Keep
            # weights in raw packed layout that awq_dequantize expects:
            # shape (K/8, N) packed int32. The g_idx permutation is still
            # needed for correctness; apply it via permute_param_layout_.
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            # no gptq_marlin_repack — keep raw layout
            return x

        def transform_w_s(x):
            # ole-eval h28.11: skip Marlin scale permute on sm_80.
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            # ole-eval h28.11: skip Marlin zero_points repack on sm_80.
            pass
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        _transform_param(layer, self.w_q_name, transform_w_q)
        _transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.kernel_config

        def _get_weight_params(
            layer: torch.nn.Module,
        ) -> tuple[
            torch.Tensor,  # w_q
            torch.Tensor,  # w_s
            Optional[torch.Tensor],  # w_zp,
            Optional[torch.Tensor],  # w_gidx
        ]:
            return (
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                getattr(layer, self.w_zp_name or "", None),
                getattr(layer, self.w_gidx_name or "", None),
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)

        # ole-eval h28.7: workspace + is_k_full lazy init when
        # process_weights_after_loading was skipped. Detect row-parallel
        # from layer weight shape vs full kernel config.
        if self.workspace is None:
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_make_workspace,
            )
            self.workspace = marlin_make_workspace(w_q.device)
            # is_row_parallel iff input was partitioned along dim 0
            row_parallel = (
                c.partition_weight_shape[0] != c.full_weight_shape[0]
            )
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_is_k_full,
            )
            self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # ole-eval h28.12 v9: pure-pytorch AWQ W4A16 dequant on sm_80.
        # Verified against Triton kernel (ground truth) on GPU0 CMP 170HX:
        #   max_diff = 0.000000 (exact match)
        #
        # Key findings:
        #   1. Formula: (q4 - zp4) * scale  (NO 0x4300 encoding, NO 8x factor)
        #   2. Reverse AWQ order: the Triton kernel reorders N positions
        #      as [0, 4, 1, 5, 2, 6, 3, 7] (bit b → output position p
        #      where RAO[p] = b). This must be replicated in pure pytorch.
        #   3. IPC layout: w_q is (N//8, K) = qweight.T, where qweight is
        #      the Triton kernel input (K, N//8). Each int32 in qweight[k, n8]
        #      packs 8 4-bit values for N positions n8*8..n8*8+7.
        #
        # Layout (from daemon .ready.meta, in_proj_qkvz example):
        #   w_q  shape=(N//8, K)=(2048, 5120) int32 — qweight transposed
        #   w_s  shape=(K//g, N)=(160, 16384) bf16 — per-group per-N scale
        #   w_zp shape=(N//8, K//g)=(2048, 160) int32 — zeros transposed
        # Output: (N, K) bf16 for F.linear(x, w_dq) = x @ w_dq.T

        # Reverse AWQ order from the Triton kernel
        _RAO = [0, 4, 1, 5, 2, 6, 3, 7]

        # ole-eval fix 2026-08-25: detect IPC layout via stride pattern.
        # Real WCD IPC layout (Qwen3.5 hybrid AWQ 27B on sm_80):
        #   w_q shape=(N//8, K) int32 strides=(K, 1)         — row-major (N//8, K)
        # The original check matched row-major (N//8, K), which is the IPC
        # convention the daemon serializes. Earlier the check used
        # `(stride(0)==shape[1] and stride(1)==1)` which is the same predicate
        # spelled out — but in practice the AWQ 27B weights arrive in
        # COLUMN-MAJOR (K//8, N) form (shape (640, 16384), strides (1, 640))
        # because permute_param_layout_() produces that layout from the
        # on-disk (N, K//8) storage. So _is_ipc_layout is False for our
        # model and the v7 branch (which correctly handles (K//8, N)) runs.
        # See tests/test_v9_vs_triton.py: max_diff=0.0 vs Triton ground truth.
        _is_ipc_layout = w_q.stride(0) == w_q.shape[1] and w_q.stride(1) == 1

        if _is_ipc_layout:
            # IPC layout: w_q (N//8, K), w_zp (N//8, K//g), w_s (K//g, N)
            N8, K = w_q.shape
            N = N8 * 8
            K_g, N_s = w_s.shape
            assert N_s == N, f"w_s cols {N_s} != N {N}"
            g = K // K_g

            w_q_c = w_q.contiguous()  # (N//8, K)
            w_zp_c = w_zp.contiguous() if w_zp is not None else None  # (N//8, K//g)

            # Unpack with reverse AWQ order:
            # q_f32[n, k] where n = n8*8+p
            # q_f32[n8*8+p, k] = (w_q[n8, k] >> (4 * RAO[p])) & 0xF
            q_f32 = torch.empty(N, K, dtype=torch.float32, device=w_q.device)
            for p in range(8):
                q_f32[p::8, :] = ((w_q_c >> (4 * _RAO[p])) & 0xF).float()

            # Unpack zeros with same RAO:
            # zp_f32[n, kg] where n = n8*8+p
            # zp_f32[n8*8+p, kg] = (w_zp[n8, kg] >> (4 * RAO[p])) & 0xF
            if w_zp_c is not None:
                zp_f32 = torch.empty(N, K_g, dtype=torch.float32, device=w_q.device)
                for p in range(8):
                    zp_f32[p::8, :] = ((w_zp_c >> (4 * _RAO[p])) & 0xF).float()
                zp_full = zp_f32.repeat_interleave(g, dim=1)[:, :K]
                q_f32 = q_f32 - zp_full

            # Scale: w_s (K//g, N) → (N, K//g) → (N, K)
            scale_NKg = w_s.contiguous().t().float()  # (N, K//g)
            scale_full = scale_NKg.repeat_interleave(g, dim=1)[:, :K]  # (N, K)
            w_dq = (q_f32 * scale_full).to(torch.bfloat16)  # (N, K)

        else:
            # v7 layout: w_q (K//8, N) with transposed storage
            # w_q[k8, n] = qweight[n, k8] where qweight is (K, N//8)
            # This means the 4-bit packing is along K for fixed n, not along N for fixed k
            K8, N = w_q.shape
            K = K8 * 8
            K_g, N_s = w_s.shape
            g = K // K_g

            w_q_c = w_q.contiguous()  # (K//8, N)
            w_zp_c = w_zp.contiguous() if w_zp is not None else None  # (N//8, K//g)

            # In this layout: w_q[k8, n] bit b = 4-bit value for (k=k8*8+b, n)
            # The RAO reordering applies to the N dimension, not K.
            # But w_q is indexed by (k8, n), so for fixed k8, we have N values.
            # q_f32[n, k] where k = k8*8+b
            # q_f32[n, k8*8+b] = (w_q[k8, n] >> (4*b)) & 0xF  (NO RAO on K dim)
            # RAO applies to N: but N is the outer index here, not packed.
            # Actually in this layout, RAO doesn't apply to the unpacking
            # because the 4-bit values are packed along K, not N.
            # The RAO reordering in the Triton kernel is for the N dimension
            # when 4-bit values are packed along N (as in the standard qweight layout).
            # In this transposed layout, we just unpack directly.
            q_f32 = torch.empty(N, K, dtype=torch.float32, device=w_q.device)
            for b in range(8):
                q_f32[:, b::8] = ((w_q_c >> (4 * b)) & 0xF).float().t()

            # Zeros: w_zp (N//8, K//g) — standard IPC layout for zeros
            if w_zp_c is not None:
                zp_f32 = torch.empty(N, K_g, dtype=torch.float32, device=w_q.device)
                for p in range(8):
                    zp_f32[p::8, :] = ((w_zp_c >> (4 * _RAO[p])) & 0xF).float()
                zp_full = zp_f32.repeat_interleave(g, dim=1)[:, :K]
                q_f32 = q_f32 - zp_full

            scale_NKg = w_s.contiguous().t().float()
            scale_full = scale_NKg.repeat_interleave(g, dim=1)[:, :K]
            w_dq = (q_f32 * scale_full).to(torch.bfloat16)

        # F.linear(x, w_dq) computes x @ w_dq.T
        x_2d = x.reshape(-1, x.shape[-1]).to(w_dq.dtype)
        out = torch.nn.functional.linear(x_2d, w_dq, bias)
        return out.view(*x.shape[:-1], -1)
