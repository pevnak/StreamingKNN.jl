# Benchmark the exact engine on a prefix of a real data file and append a row to bench/perflog.csv.
#
#   julia --project -t auto bench/perflog.jl [--data bench/knntest_1.jls] [--n 100000]
#                                            [--metric ip|cos|l2] [--k 10] [--batchsize 4096]
include(joinpath(@__DIR__, "common.jl"))

opts = parse_args(Dict("data" => joinpath(@__DIR__, "knntest_1.jls"), "n" => "100000",
                       "metric" => "ip", "k" => "10", "batchsize" => "4096"))
n = parse(Int, opts["n"]); k = parse(Int, opts["k"]); batchsize = parse(Int, opts["batchsize"])
metric = metric_from_name(opts["metric"])

println("commit $(git_commit()) on $(gethostname()), Julia $(VERSION), $(Threads.nthreads()) threads, BLAS $(BLAS.get_num_threads())")
X = load_data(opts["data"])
n = min(n, size(X, 2))
Xp = view(X, :, 1:n)
dim = size(X, 1)

println("warm-up ..."); flush(stdout)
let db = KNNDatabase(dim; metric, k)
    stream_knn!(db, view(X, :, 1:min(n, 2 * batchsize)); batchsize)
end

println("stream_knn! over $n vectors, dim $dim, metric $(metric_name(metric)), k $k, batchsize $batchsize")
db = KNNDatabase(dim; metric, k)
_, _, elapsed, flops = run_stream(db, Xp; k, batchsize)
@printf("stream: %.1f s, %.1f µs/vector, %d unique of %d, %.0f GFLOP/s\n",
        elapsed, 1e6 * elapsed / n, length(db), n, flops / elapsed / 1e9)
single_ms, batch_us = query_latency(db, X; k)
@printf("query over %d unique vectors: single %.1f ms, batched %.1f µs/query\n", length(db), single_ms, batch_us)

append_perflog(; kind = "prefix", data = opts["data"], dim, n, unique = length(db), metric, k, batchsize,
               stream_s = elapsed, flops, single_ms, batch_us)
