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
        if not hasattr(self, '_dbg_printed'):
            import os
            dbg_path = '/tmp/wcd_awq_test/layer_shapes.log'
            with open(dbg_path, 'a') as f:
                f.write(f"qshape={tuple(w_q.shape)} strides={tuple(w_q.stride())} sshape={tuple(w_s.shape)} zshape={tuple(w_zp.shape) if w_zp is not None else None} zstrides={tuple(w_zp.stride()) if w_zp is not None else None}\n")
            self._dbg_printed = True
        if not hasattr(self, '_dbg_printed'):
            print(f"[WCD-DEBUG] w_q.shape={tuple(w_q.shape)} w_s.shape={tuple(w_s.shape)} w_zp.shape={tuple(w_zp.shape) if w_zp is not None else None}", flush=True)
            self._dbg_printed = True

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

        # ole-eval h28.11-h28.12 v7: pure-pytorch AWQ W4A16 dequant on sm_80.
        # Layout (verified empirically via daemon .ready.meta + tensor strides):
        #   w_q  shape=(K/8, N)=(640, 16384) int32, strides=(1, 640) — Marlin
        #         transposed storage. Each int32 packs 8 4-bit values along K
        #         (k = k8*8..k8*8+7) for fixed n. Total 4-bit = K*N.
        #   w_s  shape=(K/g, N)=(160, 16384) bf16 — per-N scale, K/g groups
        #   w_zp shape=(N/8, K/g)=(2048, 160) int32, strides=(160, 1) — also
        #         transposed. Each int32 packs 8 4-bit zp values along K/g.
        # Output: (K, N) bf16 weight for F.linear (which computes x @ w.T).
        K_8, N = w_q.shape  # K/8 rows, N cols (logical)
        K = K_8 * 8
        K_g, _ = w_s.shape
        g = K // K_g  # group_size
        # 1) Unpack w_q: (K/8, N) int32 -> (K/8, N, 8) 4-bit -> (K, N)
        w_q_c = w_q.contiguous()  # (K/8, N) row-major copy
        shifts = torch.arange(8, device=w_q.device, dtype=torch.int32) * 4
        # (K/8, N, 8) — last dim = 8 4-bit values, but they correspond to
        # K positions k8*8..k8*8+7 for fixed (k8, n), NOT bit 0..7 in N.
        # So (K/8, N, 8) means result[k8*8+bit, n] = (K/8, N, 8)[k8, n, bit].
        # Permute to (K, N) by combining K/8 + 8 = K.
        q4 = ((w_q_c.unsqueeze(-1) >> shifts) & 0xF)  # (K/8, N, 8) int32
        # (K/8, N, 8) -> (K, N) requires merging first two dims. The (K/8, 8) part
        # combines to K with bit being inner (last 8 in K); N stays.
        q4 = q4.permute(0, 2, 1).reshape(K, N)  # (K/8, 8, N) -> (K, N)
        # Wait — (K/8, N, 8) permute(0, 2, 1) = (K/8, 8, N), then reshape (K, N)
        # requires the (K/8, 8) part to merge into K. The 8 IS the inner K.
        # After permute: outer (K/8), middle (8 = K positions), inner (N)
        # Reshape to (K, N): outer K/8 * 8 = K (K positions), inner N
        # But the order: K_8=0, K=0..7, N=0; K_8=0, K=0..7, N=1; ... K_8=1, K=0..7, N=0
        # This is wrong! N should be the slow axis.
        # Need permute(1, 0, 2): (N, K/8, 8) -> reshape (N, K) -> .t() to (K, N)
        # Simpler: just use the unpack index directly
        # ole-eval h28.12 v7: pure-pytorch AWQ W4A16 dequant on sm_80.
        # w_q shape (K/8, N)=(640, 16384) int32, strides=(1, 640) — Marlin
        # transposed storage. Each int32 packs 8 4-bit values along K direction
        # (verified empirically: kernel-side bit order is little-endian).
        # Output: (N, K) bf16 weight for F.linear.
        q_bf16 = torch.empty(N, K, dtype=torch.bfloat16, device=w_q.device)
        for bit in range(8):
            # q_bf16[:, bit::8] selects K/8 cols starting at col=bit
            # = (N, K/8) shape
            q_bf16[:, bit::8] = (((w_q_c >> (4 * bit)) & 0xF).to(torch.bfloat16).t())
        # 2) Unpack w_zp: (N/8, K/g) int32 -> (N, K/g) bf16
        #    w_zp shape (N/8, K/g) means each int32 holds 8 N positions (n=n8*8..n8*8+7)
        #    for fixed K/g position. So bit=0..7 of int32 -> N position n8*8+bit.
        #    Total 4-bit values: (N/8 * K/g) * 8 = N * K/g ✓
        #    To build (N, K_g) bf16: per (n, kg), 4-bit at (n//8, kg) bit n%8
        if w_zp is not None:
            w_zp_c = w_zp.contiguous()  # (N/8, K/g) row-major copy
            zp_bf16 = torch.empty(N, K_g, dtype=torch.bfloat16, device=w_q.device)
            for bit in range(8):
                # zp_bf16[bit::8, :] gets N/8 rows starting at row `bit`
                # = (N/8, K_g) shape. RHS: (w_zp_c >> 4*bit) & 0xF shape (N/8, K_g)
                zp_bf16[bit::8, :] = ((w_zp_c >> (4 * bit)) & 0xF).to(torch.bfloat16)
            # broadcast (N, K/g) -> (N, K) by repeat_interleave g times along K
            zp_full = zp_bf16.repeat_interleave(g, dim=1)[:, :K]  # (N, K)
        else:
            zp_full = None
        # 3) Apply zero point. AWQ asymmetric: w = (q - zp) * scale.
        #    Try both with and without -8 to see which produces valid output.
        if zp_full is not None:
            q_bf16 = q_bf16 - zp_full  # no -8 first; garbage output means we need different unpacking
        # else: q_bf16 unchanged
        # 4) Apply scale: w_s is (K/g, N) bf16. Transpose to (N, K/g), repeat g times.
        #    Then to (N, K) to match q_bf16
        scale_NKg = w_s.contiguous().t()  # (N, K/g)
        scale_full = scale_NKg.repeat_interleave(g, dim=1)[:, :K]  # (N, K)
        w_dq = (q_bf16 * scale_full).to(torch.bfloat16)  # (N, K) bf16
        # 5) F.linear(x, w_dq) computes x @ w_dq.T
        x_2d = x.reshape(-1, x.shape[-1]).to(w_dq.dtype)
        out = torch.nn.functional.linear(x_2d, w_dq, bias)
        return out.view(*x.shape[:-1], -1)

        # NOTE: h28.12 v3 was used in commit 5b806364 and works for the
        # standard Marlin-style (K/8, N) layout. Qwen3.5 hybrid mamba
        # in_proj_qkvz uses a transposed (N/8, K) layout that v3 does
        # not handle — see h28.12 v4/v5/v6 attempts in commit history.
        # For mamba hybrid AWQ, the upstream `awq_dequantize` kernel
        # needs a layout adapter. Tracking as future work in
        # docs/adr/0001-VERIFICATION_20260824_GPU0.md §6.
