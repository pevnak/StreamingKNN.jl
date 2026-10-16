# Compute and store the exact causal k-NN of a data file, for later recall evaluation of
# approximate methods, and log the run in bench/perflog.csv.
#
#   julia --project -t auto bench/groundtruth.jl [--data bench/knntest_1.jls] [--n all]
#                                                [--metric ip|cos|l2] [--k 100] [--batchsize 4096]
#
# Output: bench/groundtruth/<data>_<metric>_k<k>_n<n>.jls holding a NamedTuple with
#   ids::Matrix{Int32}      k × n, representative stream ids (column index of the data), 0-padded
#   scores::Matrix{Float32} k × n
#   dups::Vector{Int32}     for every column, the representative id of its vector (itself if unique)
#   metric, k, n, dim, data, commit, elapsed_s
# Evaluate an approximate result `pred` with `recall(gt.ids, pred; k)`.
include(joinpath(@__DIR__, "common.jl"))

opts = parse_args(Dict("data" => joinpath(@__DIR__, "knntest_1.jls"), "n" => "all",
                       "metric" => "ip", "k" => "100", "batchsize" => "4096"))
k = parse(Int, opts["k"]); batchsize = parse(Int, opts["batchsize"])
metric = metric_from_name(opts["metric"])

println("commit $(git_commit()) on $(gethostname()), Julia $(VERSION), $(Threads.nthreads()) threads, BLAS $(BLAS.get_num_threads())")
X = load_data(opts["data"])
n = opts["n"] == "all" ? size(X, 2) : min(parse(Int, opts["n"]), size(X, 2))
dim = size(X, 1)
name = splitext(basename(opts["data"]))[1]
out = joinpath(@__DIR__, "groundtruth", "$(name)_$(metric_name(metric))_k$(k)_n$(n).jls")
println("ground truth for $n vectors, dim $dim, metric $(metric_name(metric)), k $k -> $out")

db = KNNDatabase(dim; metric, k)
ids, scores, elapsed, flops = run_stream(db, view(X, :, 1:n); k, batchsize)
dups = Int32[representative(db, id) for id in 1:n]
@printf("done: %.1f s, %d unique of %d, %.0f GFLOP/s\n", elapsed, length(db), n, flops / elapsed / 1e9)

serialize(out, (ids = Int32.(ids), scores, dups, metric = metric_name(metric), k, n, dim,
                data = basename(opts["data"]), commit = git_commit(), elapsed_s = elapsed))
println("saved $(out) ($(round(filesize(out) / 2^20, digits = 1)) MB)")

single_ms, batch_us = query_latency(db, X; k)
append_perflog(; kind = "groundtruth", data = opts["data"], dim, n, unique = length(db), metric, k, batchsize,
               stream_s = elapsed, flops, single_ms, batch_us)
