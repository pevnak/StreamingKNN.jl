# Benchmark the exact engine on a prefix of a real data file and append a row to benchmark/perflog.csv.
#
#   julia --project -t auto benchmark/perflog.jl [--data benchmark/knntest_1.jls] [--n 100000]
#                                            [--metric ip|cos|l2] [--k 10] [--batchsize 4096]
#                                            [--filter 0|<gemm>|<gemm>:<head>] [--pca-sample 50000] [--check-gt <file>]
# --check-gt compares the results with a ground-truth file written by groundtruth.jl (same data, same metric).
# --filter G enables HeadFilter with a G-dimensional PCA head; --filter G:H stores an H-dimensional head and
# bounds the survivors of the G-dimensional matrix product with the remaining H-G coordinates (two stages).
include(joinpath(@__DIR__, "common.jl"))

opts = parse_args(Dict("data" => joinpath(@__DIR__, "knntest_1.jls"), "n" => "100000",
                       "metric" => "ip", "k" => "10", "batchsize" => "4096",
                       "filter" => "0", "pca-sample" => "50000", "check-gt" => "", "coarse" => "-1", "ncold" => "128"))
n = parse(Int, opts["n"]); k = parse(Int, opts["k"]); batchsize = parse(Int, opts["batchsize"])
metric = metric_from_name(opts["metric"])
fparts = parse.(Int, split(opts["filter"], ':'))
ndims = fparts[1]; headdims = length(fparts) > 1 ? fparts[2] : ndims

println("commit $(git_commit()) on $(gethostname()), Julia $(VERSION), $(Threads.nthreads()) threads, BLAS $(BLAS.get_num_threads())")
X = load_data(opts["data"])
n = min(n, size(X, 2))
Xp = view(X, :, 1:n)
dim = size(X, 1)

filter = NoFilter()
if ndims > 0
    nsample = min(parse(Int, opts["pca-sample"]), n)
    print("learning a $headdims-dimensional PCA basis on $nsample columns ... "); flush(stdout)
    tb = @elapsed basis = pca_basis(view(X, :, 1:nsample), headdims)
    @printf("%.1f s, head energy %.3f (gemm part %.3f)\n", tb, head_energy(Xp, headdims; basis), head_energy(Xp, ndims; basis = basis[:, 1:ndims]))
    coarse = parse(Int, opts["coarse"])
    global filter = coarse < 0 ? HeadFilter(basis; gemm = ndims, ncold = parse(Int, opts["ncold"])) :
                                 HeadFilter(basis; gemm = ndims, coarse, ncold = parse(Int, opts["ncold"]))
end

println("warm-up ..."); flush(stdout)
let db = KNNDatabase(dim; metric, k, filter)
    stream_knn!(db, view(X, :, 1:min(n, 2 * batchsize)); batchsize)
end

println("stream_knn! over $n vectors, dim $dim, metric $(metric_name(metric)), k $k, batchsize $batchsize, filter $(ndims > 0 ? ndims : "none")")
db = KNNDatabase(dim; metric, k, filter)
ids, scores, elapsed, flops = run_stream(db, Xp; k, batchsize)
@printf("stream: %.1f s, %.1f µs/vector, %d unique of %d, %.0f GFLOP/s equivalent\n",
        elapsed, 1e6 * elapsed / n, length(db), n, flops / elapsed / 1e9)
ndims > 0 && println("filter: ", filter_stats(db))
if !isempty(opts["check-gt"])
    gt = deserialize(opts["check-gt"])
    gt.metric == metric_name(metric) || error("ground truth metric $(gt.metric) differs from $(metric_name(metric))")
    gt.n >= n && gt.k >= k || error("ground truth covers n=$(gt.n), k=$(gt.k)")
    gids = Int.(gt.ids[1:k, 1:n]); gsc = gt.scores[1:k, 1:n]
    r_exact = recall(gids, ids; k)
    r_ties = recall(gids, gsc, ids, scores; k, metric)
    maxdiff = maximum(abs(a - b) for (a, b) in zip(scores, gsc) if !isnan(a) && !isnan(b); init = 0.0)
    @printf("check against %s: recall@%d = %.6f (ties forgiven: %.6f), max |score diff| = %.2e\n",
            basename(opts["check-gt"]), k, r_exact, r_ties, maxdiff)
end
single_ms, batch_us = query_latency(db, X; k)
@printf("query over %d unique vectors: single %.1f ms, batched %.1f µs/query\n", length(db), single_ms, batch_us)

append_perflog(; kind = ndims > 0 ? "prefix+filter$(opts["filter"])" * (opts["coarse"] == "-1" ? "" : "c$(opts["coarse"])") : "prefix", data = opts["data"], dim, n, unique = length(db), metric, k, batchsize,
               stream_s = elapsed, flops, single_ms, batch_us)
