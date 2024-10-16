import torch
import torch.nn as nn
import time

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU 메모리 사용량 측정 함수
def get_gpu_memory():
    return torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB 단위로 변환

# GPU 메모리 초기화 함수
def reset_gpu_memory():
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

# 모델 비교 테스트 함수
def compare_models(model_a, model_b, input_dim, batch_size, num_iterations=100):
    # 모델을 GPU로 이동
    model_a.to(device)
    model_b.to(device)

    # 모델을 평가 모드로 설정
    model_a.eval()
    model_b.eval()

    # 더미 입력 데이터 생성
    x = torch.randn(batch_size, input_dim).to(device)

    # 워밍업
    with torch.no_grad():
        model_a(x)
        model_b(x)

    # GPU 메모리 초기화
    reset_gpu_memory()

    # 모델 A (Combined) 측정
    start_event_a = torch.cuda.Event(enable_timing=True)
    end_event_a = torch.cuda.Event(enable_timing=True)

    start_event_a.record()
    with torch.no_grad():
        for _ in range(num_iterations):
            output_a = model_a(x)
    end_event_a.record()
    torch.cuda.synchronize()
    elapsed_time_a = start_event_a.elapsed_time(end_event_a)  # ms 단위
    peak_mem_a = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB 단위

    # GPU 메모리 초기화
    reset_gpu_memory()

    # 모델 B (Sparse) 측정
    start_event_b = torch.cuda.Event(enable_timing=True)
    end_event_b = torch.cuda.Event(enable_timing=True)

    start_event_b.record()
    with torch.no_grad():
        for _ in range(num_iterations):
            output_b = model_b(x)
    end_event_b.record()
    torch.cuda.synchronize()
    elapsed_time_b = start_event_b.elapsed_time(end_event_b)  # ms 단위
    peak_mem_b = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB 단위

    # 결과 출력
    print(f"Model A :")
    print(f"  Total Time for {num_iterations} iterations: {elapsed_time_a:.2f} ms")
    print(f"  Average Time per 1iter: {elapsed_time_a/num_iterations:.2f}ms")
    print(f"  Peak GPU Memory Usage: {peak_mem_a:.2f} MB\n")

    print(f"Model B :")
    print(f"  Total Time for {num_iterations} iterations: {elapsed_time_b:.2f} ms")
    print(f"  Average Time per 1iter: {elapsed_time_b/num_iterations:.2f}ms")
    print(f"  Peak GPU Memory Usage: {peak_mem_b:.2f} MB\n")

