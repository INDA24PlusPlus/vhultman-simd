# Building
'zig build -Dscalar=true -Doptimize=ReleaseFast' to build the scalar version.
'zig build -Dscalar=false -Doptimize=ReleaseFast' to build the vectorized version.

# Comparison
A quick comparison between the two versions when ray tracing a 8k (7680x4320) simple image using the [hyperfine](https://github.com/sharkdp/hyperfine) CLI tool.
![out_](https://github.com/user-attachments/assets/4e4dea80-c8fc-4b85-a25b-dc9d50948251)

The scalar versions traces 1 ray at a time and uses no user specified SIMD. Though the compiler is free to auto vectorize.

The vectorized versions traces 8 rays at a time through user specified SIMD (Zig's [@Vector](https://ziglang.org/documentation/0.13.0/#Vector) types)

## Scalar
Time (mean ± σ):      1.022 s ±  0.017 s    [User: 0.191 s, System: 0.034 s]
Range (min … max):    1.003 s …  1.064 s    10 runs

## Vectorized
Time (mean ± σ):     281.6 ms ±   1.5 ms    [User: 88.8 ms, System: 50.0 ms]
Range (min … max):   279.4 ms … 284.1 ms    10 runs
