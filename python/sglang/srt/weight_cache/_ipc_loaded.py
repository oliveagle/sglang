# SPDX-License-Identifier: Apache-2.0
"""ole-eval ADR-0001 lxm.2.3: ipc_loaded flag for zero-copy invariant.

A contextvars flag that the IPC model loader sets when it completes loading
weights via CUDA IPC. The DefaultModelLoader.load_weights_and_postprocess()
checks this flag and, if set, skips the
``quant_method.process_weights_after_loading(module)`` post-processing loop.

Why this is needed:
- The daemon already ran ``process_weights_after_loading`` before exporting
  IPC handles. Re-running it on the client would double-repack (e.g. AWQ
  tile re-pack, GPTQ permute) and silently corrupt the tensor data.
- Upstream IpcModelLoader already documents the skip in a code comment
  but doesn't enforce it. This flag turns the implicit skip into an
  explicit cross-module contract that any future loader refactor must
  honor.

Usage:
    from sglang.srt.weight_cache._ipc_loaded import (
        set_ipc_loaded, is_ipc_loaded, reset_ipc_loaded,
    )

    token = set_ipc_loaded()        # caller is now IPC mode
    try:
        # do IPC loading
        ...
    finally:
        reset_ipc_loaded(token)    # restore previous state
"""
from contextvars import ContextVar

_ipc_loaded_var: ContextVar = ContextVar("ipc_loaded", default=False)


def set_ipc_loaded() -> object:
    """Mark the current context as IPC-loaded. Returns a token for reset."""
    return _ipc_loaded_var.set(True)


def is_ipc_loaded() -> bool:
    """Return True if the current context is IPC-loaded."""
    return _ipc_loaded_var.get()


def reset_ipc_loaded(token: object) -> None:
    """Restore the previous IPC-loaded state using the token from set_ipc_loaded()."""
    _ipc_loaded_var.reset(token)
