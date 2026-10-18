# Shared helpers for the benchmark scripts.
using StreamingKNN
using Serialization
using LinearAlgebra
using Printf
using Dates

const PERFLOG = joinpath(@__DIR__, "perflog.csv")
const PERFLOG_HEADER = "date,commit,host,julia,threads,blas_threads,kind,data,dim,n,unique,metric,k,batchsize,stream_s,us_per_vector,gflops,single_query_ms,batch_query_us"

function git_commit()
    try
        c = strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $(@__DIR__) status --porcelain --untracked-files=no`, String)))
        dirty ? c * "-dirty" : c
    catch
        "unknown"
    end
end

metric_from_name(s::AbstractString) =
    s == "ip" ? InnerProduct() : s == "cos" ? Cosine() : s == "l2" ? Euclidean() :
    error("unknown metric $s (use ip, cos or l2)")
metric_name(::InnerProduct) = "ip"
metric_name(::Cosine) = "cos"
metric_name(::Euclidean) = "l2"

"Parse `--key value` pairs with defaults."
function parse_args(defaults::Dict{String,String})
    opts = copy(defaults)
    i = 1
    while i <= length(ARGS)
        k = ARGS[i]
        startswith(k, "--") || error("expected --option, got $k")
        k = k[3:end]
        haskey(opts, k) || error("unknown option --$k; known: $(join(sort!(collect(keys(opts))), ", "))")
        opts[k] = ARGS[i + 1]
        i += 2
    end
    opts
end

function load_data(path)
    print("loading $path ... "); flush(stdout)
    t = @elapsed X = deserialize(path)
    @printf("%s %s, %.1f s\n", summary(X), eltype(X), t)
    X
end

"""
Run the causal stream batch by batch, reporting progress, and return
`(ids, scores, elapsed, flops)` where `flops` counts the actual work done
(duplicates do not touch the matrix products).
"""
function run_stream(db, X; k, batchsize, report_every = 10)
    n = size(X, 2)
    d = size(X, 1)
    ids = Matrix{Int}(undef, k, n)
    scores = Matrix{Float32}(undef, k, n)
    flops = 0.0
    t0 = time()
    nb = cld(n, batchsize)
    for (b, lo) in enumerate(1:batchsize:n)
        hi = min(lo + batchsize - 1, n)
        B = hi - lo + 1
        flops += 2.0 * B * nslots(db) * d
        i, s = stream_knn!(db, view(X, :, lo:hi); k, batchsize = B)
        ids[:, lo:hi] = i
        scores[:, lo:hi] = s
        if b % report_every == 0 || b == nb
            el = time() - t0
            # remaining work grows with the database: estimate from flops done vs. total expected
            done = hi / n
            frac = done^2                 # cost so far ∝ (fraction)² for uniform duplicates
            eta = frac > 0 ? el / frac - el : NaN
            @printf("  batch %4d/%d  %7d vectors  %7d unique  %8.1f s elapsed  ETA %7.1f s  %6.0f GFLOP/s\n",
                    b, nb, hi, length(db), el, eta, flops / el / 1e9)
            flush(stdout)
        end
    end
    ids, scores, time() - t0, flops
end

function query_latency(db, X; k)
    q = Float32.(view(X, :, 1))
    knn(db, q; k)
    t1 = minimum(@elapsed(knn(db, q; k)) for _ in 1:3)
    Qb = Float32.(view(X, :, 1:min(1024, size(X, 2))))
    knn(db, Qb; k)
    tb = @elapsed knn(db, Qb; k)
    1e3 * t1, 1e6 * tb / size(Qb, 2)
end

function append_perflog(; kind, data, dim, n, unique, metric, k, batchsize, stream_s, flops, single_ms, batch_us)
    isfile(PERFLOG) || write(PERFLOG, PERFLOG_HEADER * "\n")
    row = join([
        Dates.format(now(), "yyyy-mm-dd HH:MM"), git_commit(), gethostname(), string(VERSION),
        Threads.nthreads(), BLAS.get_num_threads(), kind, basename(data), dim, n, unique,
        metric_name(metric), k, batchsize,
        @sprintf("%.1f", stream_s), @sprintf("%.1f", 1e6 * stream_s / n), @sprintf("%.0f", flops / stream_s / 1e9),
        @sprintf("%.2f", single_ms), @sprintf("%.1f", batch_us)], ",")
    open(PERFLOG, "a") do io
        println(io, row)
    end
    println("\nappended to $(PERFLOG):\n", PERFLOG_HEADER, "\n", row)
end
