"""Triton kernels for bench validation (not wired into training pipeline)."""
from tlb.kernels._triton_utils import TRITON_AVAILABLE, cuda_triton_available
from tlb.kernels.dispatch import attention_forward, available_backends

__all__ = [
    "TRITON_AVAILABLE",
    "attention_forward",
    "available_backends",
    "cuda_triton_available",
]
