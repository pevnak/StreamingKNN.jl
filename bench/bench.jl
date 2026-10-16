# Run with:  julia --project=. -t auto bench/bench.jl [dim] [n]
using StreamingKNN
using StreamingKNN: convert_columns!, direct_scores!, chunk_as, workspace!, load_queries!, search_slots!
using LinearAlgebra
using Printf

const dim = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1024
const n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000

println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
println("dim = $dim, n = $n, storage = Float16, compute = Float32")

# --- kernel micro benchmarks on one chunk ---------------------------------
chunksize = 1 << 15
X = randn(Float16, dim, chunksize)
buf = Matrix{Float32}(undef, dim, chunksize)
convert_columns!(buf, X, chunksize)
t = @elapsed for _ in 1:5; convert_columns!(buf, X, chunksize); end
gb = 5 * dim * chunksize * (2 + 4) / 1e9
@printf("Float16 -> Float32 conversion: %.2f GB/s (read+write)\n", gb / t)

println("\nscores of one chunk ($chunksize vectors) for a batch of B queries, ms:")
println("     B     direct    convert+gemm   gemm only")
for B in (1, 2, 4, 8, 16, 32)
    Q = randn(Float32, dim, B)
    S = Matrix{Float32}(undef, chunksize, B)
    direct_scores!(S, X, chunksize, Q)
    td = @elapsed for _ in 1:3; direct_scores!(S, X, chunksize, Q); end
    mul!(S, transpose(buf), Q)
    tg = @elapsed for _ in 1:3; mul!(S, transpose(buf), Q); end
    tc = @elapsed for _ in 1:3; convert_columns!(buf, X, chunksize); mul!(S, transpose(buf), Q); end
    @printf("%6d %10.2f %13.2f %11.2f\n", B, 1e3 * td / 3, 1e3 * tc / 3, 1e3 * tg / 3)
end

# --- end-to-end streaming ---------------------------------------------------
println("\nstream_knn! on $n vectors (causal, k = 10):")
data = randn(Float16, dim, n)
for batchsize in (256, 1024, 4096)
    local db = KNNDatabase(dim)
    stream_knn!(db, view(data, :, 1:min(n, 2000)); batchsize)   # warm up / compile
    db = KNNDatabase(dim)
    local t = @elapsed stream_knn!(db, data; batchsize)
    flops = n^2 / 2 * dim * 2
    @printf("  batchsize %5d: %7.1f s total, %7.1f µs/vector, %6.0f GFLOP/s effective\n",
            batchsize, t, 1e6 * t / n, flops / t / 1e9)
end

# --- single query latency on the full database ---------------------------
db = KNNDatabase(data)
q = randn(Float32, dim)
knn(db, q)
t = @elapsed for _ in 1:5; knn(db, q); end
@printf("\nsingle-query knn over %d vectors: %.1f ms  (extrapolated to 5M: %.0f ms)\n",
        n, 1e3 * t / 5, 1e3 * t / 5 * 5_000_000 / n)
Qb = randn(Float32, dim, 1024)
knn(db, Qb)
t = @elapsed knn(db, Qb)
@printf("batch of 1024 queries over %d vectors: %.1f ms  (%.1f µs/query; 5M: %.1f ms/query)\n",
        n, 1e3 * t, 1e6 * t / 1024, 1e6 * t / 1024 * 5_000_000 / n / 1e3)
