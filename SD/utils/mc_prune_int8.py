# mc_prune_int8.py
# One-shot "Prune40% + INT8" utility for MLP-style models (e.g., TorchMLP).
# Usage:
#   from mc_prune_int8 import prune40_int8
#   model_q, report = prune40_int8(model, amount=0.40, exclude_head=True)
#   print(report)

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Optional
import copy
import io
import warnings
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Prefer new API; fall back if older PyTorch
try:
    import torch.ao.quantization as tq
except Exception:  # pragma: no cover
    import torch.quantization as tq  # type: ignore


# =========================
# ---- helpers / core ----
# =========================

def _collect_params(
    model: nn.Module,
    module_whitelist: Tuple[type, ...] = (nn.Linear,),
    prune_bias: bool = False,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[Tuple[nn.Module, str]]:
    """Collect (module, 'weight'|'bias') pairs to prune."""
    params: List[Tuple[nn.Module, str]] = []
    exclude = set(exclude_names or [])
    for name, m in model.named_modules():
        if isinstance(m, module_whitelist) and name not in exclude:
            params.append((m, "weight"))
            if prune_bias and getattr(m, "bias", None) is not None:
                params.append((m, "bias"))
    return params


def _remove_weight_norms(model: nn.Module) -> int:
    """Remove WeightNorm hooks if present to avoid conflicts with deepcopy/prune."""
    from torch.nn.utils import remove_weight_norm
    cnt = 0
    for m in model.modules():
        # most commonly weight_norm on 'weight'
        try:
            remove_weight_norm(m, name="weight")
            cnt += 1
        except Exception:
            pass
        # some codebases name differently; be tolerant
        for alt in ("weight_g", "weight_v"):
            try:
                remove_weight_norm(m, name=alt)
                cnt += 1
            except Exception:
                pass
    return cnt


@torch.no_grad()
def _tensor_sparsity(t: torch.Tensor) -> float:
    return float((t == 0).sum().item()) / float(t.numel()) if t.numel() else 0.0


def _sparsity_report(
    model: nn.Module,
    params: Optional[List[Tuple[nn.Module, str]]] = None,
) -> Dict[str, float]:
    if params is None:
        params = _collect_params(model, prune_bias=True)
    details: Dict[str, float] = {}
    zeros, total = 0, 0
    for idx, (m, pname) in enumerate(params):
        w = getattr(m, pname, None)
        if not isinstance(w, torch.Tensor):
            continue
        s = _tensor_sparsity(w)
        details[f"{m.__class__.__name__}[{idx}].{pname}"] = s
        total += w.numel()
        zeros += (w == 0).sum().item()
    details["overall_sparsity"] = (zeros / total) if total else 0.0
    return details


def _bytes_of_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.tell()


# =========================
# ---- public entry ----
# =========================

def prune40_int8(
    model: nn.Module,
    *,
    amount: float = 0.40,
    global_pruning: bool = True,
    method: str = "l1_unstructured",
    exclude_head: bool = True,
    prune_bias: bool = False,
    in_place: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    One-shot: (1) remove WeightNorm -> (2) prune -> (3) make permanent -> (4) dynamic INT8 quantize.

    Args:
        model: trained FP32 model.
        amount: target sparsity for pruning (default 0.40 == 40%).
        global_pruning: global across all Linear weights if True; else per-layer.
        method: 'l1_unstructured' (magnitude) or 'random_unstructured'.
        exclude_head: exclude module named 'head' from pruning (common for classifier head).
        prune_bias: also prune bias terms if True.
        in_place: modify the given model if True, else try to work on a deep copy (safe default).
        seed: RNG seed (relevant for random_unstructured).

    Returns:
        model_q: INT8 dynamic-quantized model (CPU-friendly).
        report: dict with sparsity and size stats.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # ---- safe deepcopy: try copy, otherwise fall back to in-place to avoid weight_norm deepcopy crash
    copy_mode = "deepcopy"
    if in_place:
        mdl = model
        copy_mode = "in_place"
    else:
        try:
            mdl = copy.deepcopy(model)
        except Exception as e:
            warnings.warn(
                f"[prune40_int8] deepcopy failed ({e}). Falling back to in-place mode; "
                f"the original model will be modified.",
                RuntimeWarning,
            )
            mdl = model
            copy_mode = "in_place"

    # 0) clean weight norm to avoid conflicts
    _remove_weight_norms(mdl)

    # 1) collect params to prune (default: all Linear weights; optionally exclude 'head')
    exclude = {"head"} if exclude_head else set()
    params = _collect_params(
        mdl,
        module_whitelist=(nn.Linear,),
        prune_bias=prune_bias,
        exclude_names=exclude,
    )
    if not params:
        raise RuntimeError("No parameters found for pruning. Check model or exclusions.")

    # 2) choose pruning method
    method_key = method.lower().strip()
    if method_key in ("l1", "l1_unstructured", "magnitude"):
        P = prune.L1Unstructured
    elif method_key in ("rand", "random", "random_unstructured"):
        P = prune.RandomUnstructured
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 3) apply pruning
    if global_pruning:
        prune.global_unstructured(params, pruning_method=P, amount=amount)
    else:
        for m, pname in params:
            P.apply(m, name=pname, amount=amount)

    # 4) sparsity report before materialization
    report = _sparsity_report(mdl, params)

    # 5) make pruning permanent (remove reparam, keep zeros)
    for m, pname in params:
        prune.remove(m, pname)

    # 6) size stats (fp32)
    fp32_bytes = _bytes_of_state_dict(mdl.state_dict())

    # 7) dynamic INT8 quantization for CPU inference
    mdl.eval()
    mdl_cpu = mdl.to("cpu")
    mdl_q = tq.quantize_dynamic(mdl_cpu, {nn.Linear}, dtype=torch.qint8)

    # 8) size stats (int8 weights live in packed format; export state_dict for rough comparison)
    int8_bytes = _bytes_of_state_dict(mdl_q.state_dict())

    # 9) enrich report
    report.update({
        "copy_mode": copy_mode,  # 'deepcopy' | 'in_place'
        "target_amount": float(amount),
        "global_pruning": bool(global_pruning),
        "method": method_key,
        "excluded_head": bool(exclude_head),
        "prune_bias": bool(prune_bias),
        "fp32_size_bytes": float(fp32_bytes),
        "int8_size_bytes": float(int8_bytes),
        "size_reduction_ratio": float(int8_bytes) / max(1.0, float(fp32_bytes)),
    })
    return mdl_q, report


# =========================
# ---- optional quick test (comment) ----
# =========================
# if __name__ == "__main__":
#     # Example (assuming your project provides _MLP_ResBN):
#     from SD.utils.model import _MLP_ResBN
#     m = _MLP_ResBN(in_dim=784, out_dim=10)  # load your trained weights here
#     # m.load_state_dict(torch.load("ckpt.pt"))
#     m_q, rep = prune40_int8(m, amount=0.40, exclude_head=True)
#     print(rep)
