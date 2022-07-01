using BenchmarkTools
using CUDA
using GPUArrays

const suite = BenchmarkGroup()

macro benchmark_repeat(f, T, dims)
    quote
        @benchmarkable CUDA.@sync($f) setup=(x = CUDA.rand($T, $(dims)...)) teardown=(CUDA.unsafe_free!(x); CUDA.reclaim())
    end
end

# Control the size of the CUDA Array to be benchmarked
n = 8

# Benchmark `repeat(x, inner=(n, 1, 1))`
s = suite["repeat-inner-row"] = BenchmarkGroup()
s[64] = @benchmark_repeat repeat(x, inner=(64 , 1, 1)) Float32 (2^n, 2^n, 2^n)
s[128] = @benchmark_repeat repeat(x, inner=(128, 1, 1)) Float32 (2^n, 2^n, 2^n)
s[256] = @benchmark_repeat repeat(x, inner=(256, 1, 1)) Float32 (2^n, 2^n, 2^n)

s = suite["repeat-inner-col"] = BenchmarkGroup()
s[64] = @benchmark_repeat repeat(x, inner=(1, 1, 64)) Float32 (2^n, 2^n, 2^n)
s[128] = @benchmark_repeat repeat(x, inner=(1, 1, 128)) Float32 (2^n, 2^n, 2^n)
s[256] = @benchmark_repeat repeat(x, inner=(1, 1, 256)) Float32 (2^n, 2^n, 2^n)

# Benchmark `repeat(x, outer=(n, 1, 1))`
s = suite["repeat-outer-row"] = BenchmarkGroup()
s[64] = @benchmark_repeat repeat(x, outer=(64 , 1, 1)) Float32 (2^n, 2^n, 2^n)
s[128] = @benchmark_repeat repeat(x, outer=(128, 1, 1)) Float32 (2^n, 2^n, 2^n)
s[256] = @benchmark_repeat repeat(x, outer=(256, 1, 1)) Float32 (2^n, 2^n, 2^n)

s = suite["repeat-outer-col"] = BenchmarkGroup()
s[64] = @benchmark_repeat repeat(x, outer=(1, 1, 64)) Float32 (2^n, 2^n, 2^n)
s[128] = @benchmark_repeat repeat(x, outer=(1, 1, 128)) Float32 (2^n, 2^n, 2^n)
s[256] = @benchmark_repeat repeat(x, outer=(1, 1, 256)) Float32 (2^n, 2^n, 2^n)
