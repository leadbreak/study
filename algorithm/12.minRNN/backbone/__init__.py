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

Usage:
    >>> from backbone import MinGRUCUDA, is_cuda_available
    >>> if is_cuda_available():
    ...     model = MinGRUCUDA(d_model=256)

Note:
    커널 가용성을 확인할 때는 is_cuda_available(), is_triton_available() 함수를 사용하세요.
    CUDA_AVAILABLE 상수는 import 시점에 False이며, 실제 가용성은 함수 호출 시 확인됩니다.

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
    # Availability Check Functions (권장)
    is_cuda_available,
    is_triton_available,
    # Availability Flags (deprecated, use functions instead)
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
    # Availability Functions (권장)
    'is_cuda_available',
    'is_triton_available',
    # Flags (deprecated)
    'TRITON_AVAILABLE',
    'CUDA_AVAILABLE',
]
