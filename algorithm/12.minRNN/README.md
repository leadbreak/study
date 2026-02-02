# MinRNN Research

Minimal Recurrent Neural Network 연구 및 구현

## Overview

이 프로젝트는 **"Were RNNs All We Needed?"** (arXiv:2410.01201) 논문의 MinGRU/MinLSTM을 구현하고,
다양한 시퀀스 모델(Mamba, Transformer)과의 성능을 비교 분석합니다.

### Key Innovation

기존 GRU/LSTM은 게이트가 현재 hidden state에 의존하여 순차적 계산이 필수였습니다.
MinGRU는 **게이트가 hidden state에 의존하지 않도록** 단순화하여 **병렬 스캔(parallel scan)** 을 가능하게 했습니다.

```
# 기존 GRU (순차적)
g_t = σ(W_g · [h_{t-1}, x_t])  # hidden state 의존
h_t = g_t * h_{t-1} + (1 - g_t) * c_t

# MinGRU (병렬화 가능)
g_t = σ(W_g · x_t)  # 입력만 의존 → 병렬 계산 가능!
h_t = g_t * h_{t-1} + (1 - g_t) * c_t
```

## Directory Structure

```
12.minRNN/
├── README.md                          # 이 문서
├── backbone/                          # 핵심 구현체
│   ├── __init__.py                    # 모듈 exports
│   └── mingru_optimized.py            # MinGRU 최적화 구현
├── data/                              # 데이터셋
│   └── input.txt                      # TinyShakespeare
├── results/                           # 실험 결과 저장
│   ├── 10_model_comparison/           # 기본 모델 비교 결과
│   └── 11_model_comparison_compiled/  # torch.compile 적용 결과
└── *.ipynb                            # 실험 노트북들
```

## Backbone Module

### 설치 및 사용

```python
from backbone import (
    # Core Models
    MinGRUCell,           # 기본 Sequential 구현
    MinGRUTriton,         # Triton 커널 (~50-100x speedup)
    MinGRUCUDA,           # CUDA C++ 커널 (~300x speedup)
    MinGRUBlock,          # Pre-Norm + Residual 블록
    MinGRUStack,          # 다층 스택
    MinGRULanguageModel,  # 언어 모델

    # Utility Modules
    RMSNorm,              # Root Mean Square Normalization
    CausalDepthWiseConv1d,# Causal Convolution
    SwiGLUFFN,            # SwiGLU Feed-Forward Network

    # Kernel Functions
    mingru_scan_sequential,  # 순차 스캔
    mingru_scan_triton,      # Triton 병렬 스캔
    mingru_scan_cuda,        # CUDA 병렬 스캔

    # Availability Check (권장)
    is_triton_available,  # Triton 커널 가용성
    is_cuda_available,    # CUDA 커널 가용성
)
```

### 커널 성능 비교

| Kernel | Speedup vs Sequential | 특징 |
|--------|----------------------|------|
| Sequential | 1x (baseline) | Python 루프, 디버깅용 |
| Triton | ~50-100x | JIT 컴파일, 설치 쉬움 |
| CUDA C++ | ~300x | 최고 성능, 자동 JIT 컴파일 |

### 모델 구조

```
MinGRULanguageModel
├── Embedding (vocab_size → d_model)
├── CausalConv1d (short-term dependencies)
├── MinGRUStack
│   └── MinGRUBlock × n_layers
│       ├── RMSNorm
│       ├── MinGRU (Triton or CUDA)
│       ├── Residual + Dropout
│       ├── RMSNorm
│       ├── SwiGLUFFN
│       └── Residual + Dropout
├── RMSNorm
└── LM Head (tied with embedding)
```

## Notebooks

### 학습 단계별 노트북

| 노트북 | 설명 |
|--------|------|
| `01.summary.ipynb` | MinGRU 논문 요약 |
| `02.step_by_step.ipynb` | 단계별 구현 가이드 |
| `03.summary.ipynb` | 추가 분석 |

### 성능 비교 실험

| 노트북 | 설명 |
|--------|------|
| `04.comparison_*.ipynb` | 초기 비교 실험 |
| `05.comparison_*.ipynb` | 다양한 설정 비교 (seq_len, optimizer) |
| `06.triton_test_*.ipynb` | Triton 커널 테스트 |
| `07.comparison_compile_*.ipynb` | torch.compile 적용 실험 |
| `08.comparison_compile_1024_w_triton.ipynb` | Triton + torch.compile |

### 최종 벤치마크

| 노트북 | 설명 |
|--------|------|
| `09.minGRU_optimized_benchmark.ipynb` | MinGRU 커널 벤치마크 (Sequential vs Triton vs CUDA) |
| `10.model_comparison.ipynb` | 4개 모델 비교 (MinGRU Triton/CUDA, Mamba, Transformer) |
| `11.model_comparison_compiled.ipynb` | torch.compile 적용 모델 비교 |

## Model Comparison

### 비교 모델

1. **MinGRU (Triton)**: Triton JIT 커널 기반
2. **MinGRU (CUDA)**: CUDA C++ 커널 기반
3. **Mamba**: State Space Model with selective scan (`mamba-ssm`)
4. **Transformer (LLaMA)**: Flash Attention + RoPE + SwiGLU

### 비교 항목

- Training/Validation Loss
- 에포크당 학습 시간
- VRAM 사용량
- 파라미터 수
- (torch.compile 적용 시) Warmup 시간, Compile 시간

### 실험 설정

```python
SEQUENCE_LENGTH = 256
HIDDEN_DIM = 128
NUM_LAYERS = 4
BATCH_SIZE = 512
EPOCHS = 20
```

## Results

실험 결과는 `results/` 폴더에 저장됩니다:

```
results/
├── 10_model_comparison/
│   ├── model_comparison.png      # 시각화 그래프
│   ├── summary.csv               # 요약 테이블
│   └── training_history.json     # 상세 학습 기록
└── 11_model_comparison_compiled/
    ├── model_comparison_compiled.png
    ├── summary.csv
    └── training_history.json
```

## Key Findings

### MinGRU 장점

1. **단순한 구조**: 게이트가 hidden state에 의존하지 않음
2. **병렬화 가능**: Parallel scan으로 O(L) → O(log L) 시간 복잡도
3. **메모리 효율**: Transformer 대비 적은 메모리 사용
4. **긴 시퀀스**: O(L) 메모리로 긴 시퀀스 처리 가능

### 성능 특성

- **학습 품질**: Transformer와 유사한 수준의 손실 달성
- **학습 속도**: CUDA 커널로 Triton 대비 2-3배 빠름
- **메모리**: Transformer 대비 낮은 VRAM 사용
- **torch.compile**: Triton 커널과 조합 시 추가 최적화 효과

## Dependencies

```bash
# Core
torch>=2.0.0
triton>=2.1.0

# Optional (for comparison)
mamba-ssm           # Mamba SSM
flash-attn          # Flash Attention

# Visualization
matplotlib
pandas
numpy
tqdm
```

## References

- **MinGRU/MinLSTM**: [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201) (arXiv:2410.01201)
- **Mamba**: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- **LLaMA**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **Official Implementation**: [BorealisAI/minRNNs](https://github.com/BorealisAI/minRNNs)

## Citation

```bibtex
@article{feng2024were,
  title={Were RNNs All We Needed?},
  author={Feng, Leo and Tung, Frederick and Hajimirsadeghi, Hossein and Mori, Greg},
  journal={arXiv preprint arXiv:2410.01201},
  year={2024}
}
```
