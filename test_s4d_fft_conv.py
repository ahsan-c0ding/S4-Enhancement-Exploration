import torch
import time
from model.s4d import S4D as S4D_FFT
from model.s4d_modified import S4D as S4D_Direct


def benchmark(device):
    print("Running S4D FFT vs Direct Benchmark...\n")

    torch.manual_seed(0)

    B = 2           # batch size
    H = 16          # d_model
    lengths = [64, 256, 1024, 4096]

    results = []

    for L in lengths:
        print(f"Testing sequence length L = {L}")

        # Input shape expected by S4D: (B, H, L)
        u = torch.randn(B, H, L).to(device)

        model_fft = S4D_FFT(d_model=H).to(device)
        model_direct = S4D_Direct(d_model=H).to(device)

        # Copy parameters to ensure fair comparison
        model_direct.load_state_dict(model_fft.state_dict())

        _ = model_fft(u)
        _ = model_direct(u)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # FFT Timing
        runs = 3
        for _ in range(runs):
            start = time.time()
            _ = model_fft(u)
            if device.type == "cuda":
                 torch.cuda.synchronize()
        fft_time = (time.time() - start)/runs

        # Direct Convolution Timing
        runs = 3
        for _ in range(runs):
            start = time.time()
            _ = model_direct(u)
            if device.type == "cuda":
                torch.cuda.synchronize()
        direct_time = (time.time() - start)/3

        results.append((L, fft_time, direct_time))

        print(
            f"L={L:4d} | FFT: {fft_time:.6f}s | Direct: {direct_time:.6f}s\n"
        )

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n")

    results = benchmark(device)

    print("\nBenchmark Summary:")
    print("Length | FFT (s) | Direct Convolution (s)")
    print("__________________________________")

    for L, fft_time, direct_time in results:
        print(f"{L:6d} | {fft_time:8.6f} | {direct_time:14.6f}")


if __name__ == "__main__":
    main()

"""
Benchmark Summary:
Length | FFT (s) | Direct Convolution (s)
__________________________________
    64 | 0.000804 |       0.000368
   256 | 0.000875 |       0.001702
  1024 | 0.002778 |       0.010298
  4096 | 0.010202 |       1.611956

The diagonal S4 (S4D) parameterization significantly reduces the number of trainable parameters from 4226 in full S4 to 194 in S4D by replacing the full 
NxN state matrix with a diagonal complex parameterization.

Benchmark results show that direct convolution and FFT-based convolution perform similarly for short sequences (L ≤ 256). 
However, as sequence length increases, the quadratic complexity of direct convolution O(L^2 * N) becomes dominant. At L = 4096, direct convolution is over 
150x slower than FFT-based convolution. This empirically confirms the theoretical complexity difference between O(L^2 *N) and O(LlogLN).
"""