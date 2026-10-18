# Streaming k-NN comparison on real data (Qwen embeddings, dim 4096, Float16 source).
#
# Identical workload for every method: process the stream in batches; each batch is first
# queried against everything inserted before it, then inserted.  (Per-column causality
# inside a batch is extra work that only StreamingKNN can do, timed separately.)
# Metric: Euclidean.  All vectors have unit norm, so this ranks like cosine / inner product.
# Recall is computed after mapping ids to the first occurrence of each distinct vector,
# because the data is ~50 % exact duplicates and libraries differ in which copy they report.
# Setup (the comparison packages are not dependencies of StreamingKNN):
#   julia --project=bench/compare -e 'using Pkg; Pkg.develop(path="."); \
#       Pkg.add(["NearestNeighbors","SimilaritySearch","HNSW","Distances","Faiss","Serialization"])'
#   julia --project=bench/compare -t auto bench/compare/compare.jl 20000 1024
include(joinpath(@__DIR__, "common.jl"))
using StreamingKNN
n  = parse(Int, get(ARGS, 1, "20000"))
bs = parse(Int, get(ARGS, 2, "1024"))
skip = length(ARGS) >= 3 ? Set(split(ARGS[3], ",")) : Set{String}()
k = 10
X16 = load_prefix(n); X32 = Float32.(X16); d = size(X32, 1)
canon = canonical_map(X16)
@printf("n = %d, dim = %d, k = %d, batch = %d, threads = %d, BLAS = %d\n", n, d, k, bs, Threads.nthreads(), BLAS.get_num_threads())
@printf("duplicates: %d of %d columns repeat an earlier vector\n\n", count(j -> canon[j] != j, 1:n), n)

const TRUTH = Ref{Matrix{Int}}()
results = Tuple{String,Float64,Float64,String}[]
function run_method(f::Function, name::String, note::String)
    name in skip && return nothing
    print(rpad(name, 28)); flush(stdout)
    local ids, t
    try
        t = @elapsed ids = f()
    catch e
        println("FAILED: ", first(split(sprint(showerror, e), "\n"))[1:min(end, 120)])
        push!(results, (name, NaN, NaN, "failed")); return nothing
    end
    r = isassigned(TRUTH) ? recall_at(canonicalise(TRUTH[], canon), canonicalise(ids, canon), k) : 1.0
    @printf("%8.1f s   recall %.4f   %s\n", t, r, note)
    push!(results, (name, t, r, note))
    serialize(joinpath(@__DIR__, "ids_$(replace(name, " " => "_")).jls"), ids)
    ids
end

"Run the shared batch loop; `query!` answers a batch, `insert!` adds it."
function batch_loop(query!, insert!)
    ids = zeros(Int, k, n)
    for lo in 1:bs:n
        hi = min(lo + bs - 1, n)
        lo > 1 && query!(ids, lo, hi)
        insert!(lo, hi)
    end
    ids
end

# ---------------------------------------------------------------- StreamingKNN (ground truth)
TRUTH[] = run_method("StreamingKNN exact", "exact, this table's ground truth") do
    db = KNNDatabase(d; metric = StreamingKNN.Euclidean(), k)
    batch_loop((ids, lo, hi) -> (ids[:, lo:hi] = first(knn(db, view(X32, :, lo:hi); k))),
               (lo, hi) -> append!(db, view(X16, :, lo:hi)))
end

run_method("StreamingKNN + filter", "exact, HeadFilter(1024, gemm=512)") do
    basis = pca_basis(X16, 1024)
    db = KNNDatabase(d; metric = StreamingKNN.Euclidean(), k, filter = HeadFilter(basis; gemm = 512))
    batch_loop((ids, lo, hi) -> (ids[:, lo:hi] = first(knn(db, view(X32, :, lo:hi); k))),
               (lo, hi) -> append!(db, view(X16, :, lo:hi)))
end

run_method("StreamingKNN per-column", "exact, strict per-column causality") do
    db = KNNDatabase(d; metric = StreamingKNN.Euclidean(), k)
    first(stream_knn!(db, X16; batchsize = bs))
end

# ---------------------------------------------------------------- SimilaritySearch
using SimilaritySearch, Distances
function ss_stream(kind)
    ctx = kind === :graph ? SearchGraphContext() : GenericContext()
    db = BlockMatrixDatabase(d, Float32)
    idx = Ref{Any}(nothing)
    Q = knnqueue(KnnSorted, k)
    batch_loop(
        (ids, lo, hi) -> begin
            for t in lo:hi
                reuse!(Q)
                search(idx[], ctx, view(X32, :, t), Q)
                @inbounds for r in 1:min(Q.ep - Q.sp + 1, k)
                    ids[r, t] = Int(Q.ids[Q.sp + r - 1])
                end
            end
        end,
        (lo, hi) -> begin
            batch = MatrixDatabase(Matrix(view(X32, :, lo:hi)))
            if idx[] === nothing
                append_items!(db, batch)
                idx[] = kind === :graph ? SearchGraph(SqEuclidean(), db) :
                                          SimilaritySearch.Exact.ParallelExhaustiveSearch(SqEuclidean(), db)
                index!(idx[], ctx)
            else
                append_items!(idx[], ctx, batch)
            end
        end)
end
run_method(() -> ss_stream(:graph), "SimilaritySearch graph", "approximate, incremental")
run_method(() -> ss_stream(:exhaustive), "SimilaritySearch exact", "exact, ParallelExhaustiveSearch")

# ---------------------------------------------------------------- HNSW.jl
using HNSW
run_method("HNSW.jl", "approximate, M=16 efC=100 ef=50") do
    data = [X32[:, i] for i in 1:n]
    h = HierarchicalNSW(data; M = 16, efConstruction = 100, ef = 50)
    batch_loop(
        (ids, lo, hi) -> begin
            idx, _ = knn_search(h, data[lo:hi], k)
            @inbounds for (c, t) in enumerate(lo:hi)
                col = idx isa AbstractMatrix ? view(idx, :, c) : idx[c]
                for r in eachindex(col); r <= k && (ids[r, t] = Int(col[r])); end
            end
        end,
        (lo, hi) -> add_to_graph!(h, lo:hi))
end

# ---------------------------------------------------------------- NearestNeighbors.jl
nn_stream(treetype) = batch_loop(
    (ids, lo, hi) -> begin
        tree = treetype(X32[:, 1:lo-1], Distances.Euclidean())
        idx, _ = NearestNeighbors.knn(tree, [X32[:, t] for t in lo:hi], min(k, lo - 1), true)
        @inbounds for (c, t) in enumerate(lo:hi), r in eachindex(idx[c]); ids[r, t] = idx[c][r]; end
    end,
    (lo, hi) -> nothing)
using NearestNeighbors
run_method(() -> nn_stream(BruteTree), "NearestNeighbors brute", "exact, index rebuilt each batch")
run_method(() -> nn_stream(KDTree), "NearestNeighbors KDTree", "exact, index rebuilt each batch")

# ---------------------------------------------------------------- Faiss
using Faiss
run_method("Faiss Flat", "exact, incremental add (via Python)") do
    fidx = Faiss.Index(d, "Flat")
    batch_loop(
        (ids, lo, hi) -> begin
            _, I = Faiss.search(fidx, permutedims(X32[:, lo:hi]), k)
            @inbounds for (c, t) in enumerate(lo:hi), r in 1:k
                v = I[c, r]; ids[r, t] = v < 1 ? 0 : Int(v)
            end
        end,
        (lo, hi) -> Faiss.add(fidx, permutedims(X32[:, lo:hi])))
end

# ---------------------------------------------------------------- summary
println("\n", rpad("method", 28), rpad("seconds", 10), rpad("vs ours", 10), rpad("recall@10", 11), "note")
base = results[1][2]
for (name, t, r, note) in results
    @printf("%-28s%-10s%-10s%-11s%s\n", name,
            isnan(t) ? "-" : @sprintf("%.1f", t), isnan(t) ? "-" : @sprintf("%.2fx", t / base),
            isnan(r) ? "-" : @sprintf("%.4f", r), note)
end
serialize(joinpath(@__DIR__, "results_n$(n).jls"), results)
