import torch
import time


def benchmark_matmul(size: int, device: torch.device) -> float:
    # Allocate two random matrices on the given device
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Actual timed run
    start = time.time()
    c = torch.mm(a, b)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    return end - start


if __name__ == "__main__":
    SIZE = 10000
    
    # CPU benchmark
    cpu_device = torch.device('cpu')
    print(f"---> Running CPU matmul for {SIZE}×{SIZE} ...")
    cpu_time = benchmark_matmul(SIZE, cpu_device)
    print(f"CPU time: {cpu_time:.2f} s\n")

    # CUDA benchmark
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        print(f"---> Running CUDA matmul for {SIZE}×{SIZE} ...")
        cuda_time = benchmark_matmul(SIZE, cuda_device)
        print(f"CUDA time: {cuda_time:.2f} s\n")
    else:
        print("CUDA not available on this machine.")

    # Summary
    if 'cuda_time' in locals():
        speedup = cpu_time / cuda_time
        print(f"Speedup (CPU ÷ CUDA): {speedup:.2f}×")