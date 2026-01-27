"""
minRNN Backbone Module
======================

Minimal Recurrent Neural Network 구현체 모음

이 모듈은 "Were RNNs All We Needed?" (arXiv:2410.01201) 논문의
MinGRU/MinLSTM 구현체와 최적화된 CUDA/Triton 커널을 제공합니다.

Models:
    - MinGRUCell: 기본 MinGRU 셀 (Sequential 구현)
    - MinGRUTriton: Triton 커널 기반 병렬 스캔
    - MinGRUCUDA: 최적화된 CUDA C++ 커널
    - MinGRULanguageModel: 언어 모델용 MinGRU 스택

References:
    - Paper: https://arxiv.org/abs/2410.01201
    - Official: https://github.com/BorealisAI/minRNNs
"""

from .mingru_optimized import (
    # Core Modules
    MinGRUCell,
    MinGRUTriton,
    MinGRUCUDA,
    MinGRUBlock,
    MinGRUStack,
    MinGRULanguageModel,
    # Utility Classes
    RMSNorm,
    CausalDepthWiseConv1d,
    SwiGLUFFN,
    # Kernel Functions
    mingru_scan_sequential,
    mingru_scan_triton,
    mingru_scan_cuda,
    # Availability Flags
    TRITON_AVAILABLE,
    CUDA_AVAILABLE,
)

__version__ = '2.0.0'
__all__ = [
    # Models
    'MinGRUCell',
    'MinGRUTriton',
    'MinGRUCUDA',
    'MinGRUBlock',
    'MinGRUStack',
    'MinGRULanguageModel',
    # Utilities
    'RMSNorm',
    'CausalDepthWiseConv1d',
    'SwiGLUFFN',
    # Functions
    'mingru_scan_sequential',
    'mingru_scan_triton',
    'mingru_scan_cuda',
    # Flags
    'TRITON_AVAILABLE',
    'CUDA_AVAILABLE',
]
