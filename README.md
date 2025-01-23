# Building

- Scalar Version:
  
  `zig build -Dscalar=true -Doptimize=ReleaseFast`
- Vectorized Version:
  
  `zig build -Dscalar=false -Doptimize=ReleaseFast`

# Results
The manually vectorized (SIMD) version is roughly 3.63x faster than the scalar version with SSE auto vectorization and roughly 10.5x faster than the scalar version with no auto vectorization.

# Performance Comparison
Below is a comparison of execution times when ray tracing an 8K image (7680x4320) using the [hyperfine](https://github.com/sharkdp/hyperfine CLI benchmarking tool.

## Resulting image:
![out_](https://github.com/user-attachments/assets/4e4dea80-c8fc-4b85-a25b-dc9d50948251)


## Differences
- Scalar version (With SSE): Traces one ray at a time and the compiler is free to auto vectorize and generate SIMD instructions as it sees fit.
- Scalar version (Without SSE): Traces one ray at a time but code generation of SSE instructions have been restricted. (Compiler appears to not generate any AVX instructions)
- Vectorized version: Traces 8 rays at a time through programmer specified SIMD (Zig's [@Vector](https://ziglang.org/documentation/0.13.0/#Vector) types)

## Detailed results
### Scalar (With SSE)
Executable built with `zig build -Doptimize=ReleaseFast -Dscalar=true -Ddisable-sse=false -Dcpu=native`
Time (mean ± σ):      1.022 s ±  0.017 s    [User: 0.191 s, System: 0.034 s]
Range (min … max):    1.003 s …  1.064 s    10 runs

### Scalar (Without SSE)
Executable built with `zig build -Doptimize=ReleaseFast -Dscalar=true -Ddisable-sse=true -Dcpu=native`
Time (mean ± σ):      2.957 s ±  0.020 s    [User: 0.192 s, System: 0.023 s]
Range (min … max):    2.936 s …  2.987 s    10 runs

### Vectorized
Executable built with `zig build -Doptimize=ReleaseFast -Dscalar=false -Ddisable-sse=false -Dcpu=native`
Time (mean ± σ):     281.6 ms ±   1.5 ms    [User: 88.8 ms, System: 50.0 ms]
Range (min … max):   279.4 ms … 284.1 ms    10 runs

# Some statistics from VTune
## Vectorized
- Time when ran in VTune: 0.288s
- IPC: 1.256
- CPI: 0.596
- SP GFLOPS: 21.843
- Vectorization: 93.7% of Packed FP ops
- Avg CPU Freq: 5.2 GHz

## Scalar version (With SSE)
- Time when ran in VTune: 1.031s
- IPC: 1.861
- CPI: 0.422
- SP GFLOPS: 6.046
- Vectorization: 11.2% of Packed FP ops
- Avg CPU Freq: 5.3 GHz

## Scalar version (no SSE)
- Time when ran in VTune: 2.916s
- IPC: 1.753
- CPI: 0.452
- SP GFLOPS: 0.010
- x87 GFLOPS: 1.636
- Vectorization: 0.0% of Packed FP ops
- Avg CPU Freq: 5.3GHz
