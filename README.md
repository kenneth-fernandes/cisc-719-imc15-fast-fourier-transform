# IMC15 — GPU-Accelerated Fast Fourier Transform

**Author:** Kenneth Peter Fernandes  
**Course:** CISC 719 — Contemporary Computing Systems Modeling Algorithms (CCSM), Harrisburg University, Spring 2026  
**Topic:** Fast Fourier Transform (Quinn Ch. 15)  
**Execution:** Google Colab (GPU runtime — T4)  

---

## Overview

Implements and benchmarks the 1D Fast Fourier Transform across five backends — from a serial O(N²) reference DFT through CPU-parallel SciPy to a custom Numba CUDA butterfly kernel — following the Cooley-Tukey radix-2 DIT algorithm described in Quinn Chapter 15.

All implementations are in a single Jupyter notebook.

---

## Repository Structure

```
cisc-719-imc15-fast-fourier-transform/
├── README.md
├── ai-log.csv                       # AI tool usage log
├── notebooks/
│   └── imc15_fft_main.ipynb         # All implementations, benchmarks, PCAM,
│                                    #   Roofline analysis, and conclusions
├── results/
│   ├── benchmarks.csv               # Timing data (all implementations × sizes × dtypes)
│   ├── benchmarks.png               # Throughput, wall-clock, and speedup plots
│   └── roofline.png                 # Roofline model plot
└── docs/
    ├── src/
    │   ├── report.tex               # Technical report (LaTeX source)
    │   └── references.bib          # Bibliography
    └── pdf/
        └── report.pdf              # Compiled report
```

---

## How to Run

1. Open `notebooks/imc15_fft_main.ipynb` in Google Colab
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells top-to-bottom (`Runtime → Run all`)
4. Results are written to `results/` (benchmarks.csv, benchmarks.png, roofline.png)

---

## Implementations

| # | Implementation | Backend | Complexity |
|---|----------------|---------|------------|
| 1 | Reference DFT | NumPy (matrix) | O(N²) — correctness oracle |
| 2 | NumPy FFT | `np.fft.fft` | O(N log N) — serial baseline |
| 3 | SciPy FFT | `scipy.fft.fft` + workers | O(N log N) — CPU parallel |
| 4 | CuPy FFT | `cp.fft.fft` (cuFFT) | O(N log N) — GPU library |
| 5 | Numba CUDA FFT | Custom butterfly kernel | O(N log N) — GPU custom |

---

## Benchmark Configuration

- **Signal sizes:** 2^10, 2^14, 2^18, 2^20, 2^22, 2^24
- **Dtypes:** complex64 (FP32) and complex128 (FP64)
- **Protocol:** 3 warmup runs + 10 timed runs, median reported
- **Throughput metric:** GFLOP/s using 5N log₂(N) FLOPs
- **GPU timing:** CUDA events (`cp.cuda.Event`)

---

## Key Results (T4 GPU, complex64, N = 2^24)

| Implementation | Time (ms) | GFLOP/s | Speedup vs NumPy |
|----------------|-----------|---------|-----------------|
| Reference DFT | 14,413 | 0.0001 | — |
| NumPy FFT | 1,110.5 | 1.81 | 1× |
| SciPy FFT | 599.9 | 3.36 | 1.9× |
| CuPy FFT | 4.30 | 468.5 | 258× |
| Numba CUDA FFT | 13,503 | 0.15 | 0.08× |

CuPy (cuFFT) achieves 468.5 GFLOP/s at N=2^24, a 258× speedup over the NumPy serial baseline. The custom Numba kernel is correct but unoptimized — no shared memory tiling or coalesced access — and runs slower than CPU at all tested sizes.

---

## References

- Quinn, M. J. (2004). *Parallel Programming in C with MPI and OpenMP*, Ch. 15.
- Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297–301.
- NVIDIA Corporation. (2024). cuFFT Library User's Guide.
- CuPy Developers. (2024). CuPy: NumPy/SciPy-compatible Array Library for GPU-accelerated Computing.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful Visual Performance Model. *CACM*, 52(4).
