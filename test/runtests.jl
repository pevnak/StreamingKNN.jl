using StreamingKNN
using StreamingKNN: reference_score, TopK, threshold, sorted_indices!, reset!
using Test
using Random
using LinearAlgebra

const METRICS = (InnerProduct(), Cosine(), Euclidean())

better(::InnerProduct) = (a, b) -> a > b
better(::Cosine) = (a, b) -> a > b
better(::Euclidean) = (a, b) -> a < b

"""
Naive reference: unique alive vectors (after conversion to T) with their representative id,
scored one by one.  `points` is a vector of (id, vector) pairs in stream order.
"""
function naive_knn(metric, points, q, k)
    scores = [(reference_score(metric, x, q), id) for (id, x) in points]
    sort!(scores; lt = (a, b) -> better(metric)(a[1], b[1]) || (a[1] == b[1] && a[2] < b[2]))
    scores[1:min(k, length(scores))]
end

"Fold the stream into (representative id, vector) pairs respecting `isequal` duplicates."
function fold_duplicates(::Type{T}, X) where {T}
    reps = Dict{Vector{T},Int}()
    out = Tuple{Int,Vector{T}}[]
    for j in axes(X, 2)
        v = Vector{T}(X[:, j])
        if !haskey(reps, v)
            reps[v] = j
            push!(out, (j, v))
        end
    end
    out
end

# Selection for Euclidean goes through the expansion formula in Float32, so candidates
# whose distances differ by less than ~1e-3 may swap; reported distances are exact.
order_atol(::Euclidean) = 2e-3
order_atol(::Metric) = 1e-5

function check_against_reference(metric, ids, scores, points_before, q, k; rtol = 1e-4)
    ref = naive_knn(metric, points_before, q, k)
    m = length(ref)
    @test all(ids[m+1:end] .== 0)
    @test all(isnan, scores[m+1:end])
    for r in 1:m
        @test isapprox(scores[r], ref[r][1]; rtol, atol = order_atol(metric))
        # the returned id must have the score it claims (ties make exact id matching too strict)
        x = only(x for (id, x) in points_before if id == ids[r])
        @test isapprox(reference_score(metric, x, q), scores[r]; rtol, atol = 1e-5)
    end
end

@testset "StreamingKNN" begin

@testset "TopK heap" begin
    h = TopK{Float32}(3)
    @test threshold(h) == -Inf32
    for (key, id) in zip(Float32[5, 1, 7, 3, 9, 2], 1:6)
        push!(h, key, id)
    end
    @test h.n == 3
    @test threshold(h) == 5f0
    p = sorted_indices!(h)
    @test h.keys[p] == Float32[9, 7, 5]
    @test h.ids[p] == [5, 3, 1]
    push!(h, -Inf32, 99)
    @test h.n == 3 && 99 ∉ h.ids
    reset!(h)
    @test h.n == 0
    push!(h, -Inf32, 1)          # -Inf must never enter, even when empty
    @test h.n == 0
end

@testset "stream == knn_push! ($(T), $(typeof(m)), dim=$d)" for T in (Float16, Float32, Float64),
        m in METRICS, d in (5, 33)
    rng = MersenneTwister(hash((T, typeof(m), d)))
    n = 150
    X = randn(rng, Float32, d, n)
    # inject duplicates: exact copies of earlier columns, also inside the same batch
    for j in (10, 11, 50, 51, 52, 149, 150)
        X[:, j] = X[:, j - 7]
    end
    X[:, 30] = X[:, 29]
    k = 7
    for batchsize in (1, 4, 13, 1000)
        db = KNNDatabase{T}(d; metric = m, k, chunksize = 16, direct_batch = 2)
        ids, scores = stream_knn!(db, X; batchsize)
        @test size(ids) == (k, n) && size(scores) == (k, n)
        @test nseen(db) == n
        folded = fold_duplicates(T, X)
        @test length(db) == length(folded)
        for t in 1:n
            before = [(id, x) for (id, x) in folded if id < t]
            check_against_reference(m, ids[:, t], scores[:, t], before, X[:, t], k)
        end
        # the final state answers plain queries over everything
        qids, qscores = knn(db, X[:, 1:20]; k, batchsize = 6)
        for t in 1:20
            check_against_reference(m, qids[:, t], qscores[:, t], folded, X[:, t], k)
        end
        # streaming one by one gives identical results
        db2 = KNNDatabase{T}(d; metric = m, k, chunksize = 16)
        for t in 1:n
            i2, s2 = knn_push!(db2, X[:, t]; k)
            @test all(isapprox.(s2, scores[:, t]; rtol = 1e-4, atol = 1e-5, nans = true))
            before = [(id, x) for (id, x) in folded if id < t]
            check_against_reference(m, i2, s2, before, X[:, t], k)
        end
    end
end

@testset "duplicates bookkeeping" begin
    db = KNNDatabase(3; metric = Euclidean())
    a = Float16[1, 2, 3]; b = Float16[4, 5, 6]
    @test push!(db, a) == 1
    @test push!(db, b) == 2
    @test push!(db, a) == 3
    @test push!(db, Float32.(a)) == 4           # same after conversion to Float16
    @test push!(db, Float32[1, 2, 3.0004]) == 5  # rounds to a in Float16 -> duplicate
    @test length(db) == 2 && nseen(db) == 5 && nslots(db) == 2
    @test duplicates(db, 1) == [1, 3, 4, 5]
    @test duplicates(db, 4) == [1, 3, 4, 5]
    @test duplicates(db, 2) == [2]
    @test representative(db, 5) == 1
    @test db[5] == a
    ids, _ = knn(db, a; k = 2)
    @test ids == [1, 2]
    # delete the representative: the next id takes over, vector stays
    delete!(db, 1)
    @test isdeleted(db, 1) && !isdeleted(db, 3)
    @test duplicates(db, 3) == [3, 4, 5]
    @test representative(db, 5) == 3
    @test length(db) == 2
    ids, _ = knn(db, a; k = 2)
    @test ids == [3, 2]
    delete!(db, 3); delete!(db, 4)
    @test duplicates(db, 5) == [5]
    @test knn(db, a; k = 1)[1] == [5]
    delete!(db, 5)
    @test length(db) == 1
    @test knn(db, a; k = 3)[1] == [2, 0, 0]
    @test_throws KeyError db[5]
    @test_throws KeyError duplicates(db, 1)
    delete!(db, 5)                               # idempotent
    @test length(db) == 1
    # re-inserting a deleted vector creates a fresh slot
    @test push!(db, a) == 6
    @test length(db) == 2 && nslots(db) == 3
    @test knn(db, a; k = 1)[1] == [6]
end

@testset "deletion and compaction ($(typeof(m)))" for m in METRICS
    rng = MersenneTwister(1)
    d, n, k = 9, 200, 5
    X = randn(rng, Float32, d, n)
    X[:, 100] = X[:, 3]
    db = KNNDatabase{Float16}(X; metric = m, k, chunksize = 32)
    todelete = [1:3; 50:70; 150; 199; 200]
    foreach(id -> delete!(db, id), todelete)
    @test length(db) == 200 - length(todelete)   # id 100 duplicates id 3; slot survives, its own id survives
    @test deadfraction(db) > 0
    folded = fold_duplicates(Float16, X)
    # alive representative for each slot after deletions
    alive_points = Tuple{Int,Vector{Float16}}[]
    for (id, x) in folded
        dupids = [j for j in 1:n if Vector{Float16}(X[:, j]) == x && j ∉ todelete]
        isempty(dupids) || push!(alive_points, (minimum(dupids), x))
    end
    Q = randn(rng, Float32, d, 15)
    ids1, sc1 = knn(db, Q; k)
    for t in 1:15
        check_against_reference(m, ids1[:, t], sc1[:, t], alive_points, Q[:, t], k)
    end
    @test !any(id -> id ∈ todelete, ids1)
    compact!(db)
    @test deadfraction(db) == 0
    @test nslots(db) == length(db)
    ids2, sc2 = knn(db, Q; k)
    @test ids2 == ids1
    @test all(isapprox.(sc2, sc1; rtol = 1e-5, nans = true))
    @test db[100] == Vector{Float16}(X[:, 3])
    @test duplicates(db, 100) == [100]
    # database keeps working after compaction: stream more, including a duplicate of a deleted one
    Y = hcat(X[:, 1], randn(rng, Float32, d, 40))
    ids3, sc3 = stream_knn!(db, Y; batchsize = 7)
    @test nseen(db) == n + 41
    @test ids3[1, 2] == n + 1 || sc3[1, 2] isa Real   # trivially well-formed
    before = copy(alive_points)
    for t in 1:41
        check_against_reference(m, ids3[:, t], sc3[:, t], before, Y[:, t], k)
        push!(before, (n + t, Vector{Float16}(Y[:, t])))
    end
    compact!(db)
    @test length(db) == nslots(db)
end

@testset "initialised database, odd shapes" begin
    rng = MersenneTwister(7)
    X0 = randn(rng, Float16, 6, 70)
    db = KNNDatabase(X0; metric = Cosine(), chunksize = 8, k = 3)
    @test StreamingKNN.storagetype(db) === Float16
    @test length(db) == 70
    ids, sc = stream_knn!(db, randn(rng, Float16, 6, 5); batchsize = 2)
    @test size(ids) == (3, 5)
    @test all(1 .<= ids .<= 75)
    @test all(-1 .<= sc .<= 1)
    @test_throws DimensionMismatch knn(db, randn(5))
    @test_throws DimensionMismatch push!(db, randn(7))
    # empty database
    e = KNNDatabase(4)
    ids, sc = knn(e, randn(4); k = 3)
    @test ids == [0, 0, 0] && all(isnan, sc)
    ids, sc = knn_push!(e, randn(4); k = 2)
    @test ids == [0, 0]
    ids, sc = knn_push!(e, randn(4); k = 2)
    @test ids == [1, 0]
    # k larger than the database
    ids, _ = knn(e, randn(4); k = 10)
    @test count(!=(0), ids) == 2
    # zero vectors with cosine do not produce NaN
    z = KNNDatabase(3; metric = Cosine())
    push!(z, zeros(3)); push!(z, [1.0, 0, 0])
    ids, sc = knn(z, zeros(3); k = 2)
    @test all(sc .== 0)
    ids, sc = knn(z, [1.0, 0, 0]; k = 2)
    @test ids == [2, 1] && sc[1] ≈ 1 && sc[2] == 0
end

"Data with a decaying spectrum in a random orientation plus near-duplicates, so the filter prunes."
function spectral_data(rng, d, n; decay = 1.4)
    U = Matrix(qr(randn(rng, d, d)).Q)
    X = Float32.(U * (randn(rng, Float32, d, n) .* Float32.(decay .^ (-(0:d-1)))))
    for j in 5:7:n                                   # near-duplicates and exact duplicates
        X[:, j] = X[:, j - 3] .+ Float32(1e-3) .* randn(rng, Float32, d)
    end
    for j in 9:11:n
        X[:, j] = X[:, j - 4]
    end
    X
end

@testset "HeadFilter gives identical results ($(T), $(typeof(m)), $(mode))" for T in (Float16, Float32),
        m in METRICS, mode in (:pca, :pca2, :prefix)
    rng = MersenneTwister(hash((T, typeof(m), mode)))
    d, n, k = 48, 1200, 7
    X = spectral_data(rng, d, n)
    if mode == :prefix
        X = Float32.(Matrix(qr(randn(rng, d, d)).Q)' * X)   # rotate so the energy is spread, prefix prunes little
        X = Float32.(pca_basis(X, d)' * X)                   # ... then align the data with its own PCA axes
    end
    f = mode == :pca  ? HeadFilter(pca_basis(X[:, 1:400], 6); coarse = 2, ncold = 8) :
        mode == :pca2 ? HeadFilter(pca_basis(X[:, 1:400], 12); gemm = 6, coarse = 3, ncold = 8) :
                        HeadFilter(8; gemm = 6, coarse = 0)
    @test head_energy(X, 6; basis = mode == :prefix ? nothing : f.basis[:, 1:6]) > 0.9
    ref = KNNDatabase{T}(d; metric = m, k, chunksize = 128)
    rids, rsc = stream_knn!(ref, X; batchsize = 300)
    for batchsize in (1, 300)
        db = KNNDatabase{T}(d; metric = m, k, filter = f, chunksize = 128)
        ids, sc = stream_knn!(db, X; batchsize)
        @test all(isapprox.(sc, rsc; rtol = 1e-4, atol = 1e-5, nans = true))
        # ids may differ only where scores tie
        for t in 1:n, r in 1:k
            ids[r, t] == rids[r, t] || @test isapprox(sc[r, t], rsc[r, t]; rtol = 1e-4, atol = 1e-5)
        end
        st = filter_stats(db)
        @test st.scanned > 0 && st.refined < 0.5 * st.scanned
        mode == :pca || @test 0 < st.stage2 < st.scanned
        # plain queries over the final database
        qids, qsc = knn(db, X[:, 1:50]; k, batchsize = 20)
        rq = knn(ref, X[:, 1:50]; k)
        @test all(isapprox.(qsc, rq[2]; rtol = 1e-4, atol = 1e-5, nans = true))
        # k larger than the number of stored vectors and an empty database
        e = KNNDatabase{T}(d; metric = m, filter = f)
        @test knn(e, X[:, 1]; k = 3)[1] == [0, 0, 0]
        push!(e, X[:, 1]); push!(e, X[:, 2])
        i2, s2 = knn(e, X[:, 3]; k = 5)
        @test count(!=(0), i2) == 2 && count(!isnan, s2) == 2
        @test isapprox(s2[1], reference_score(m, Vector{T}(X[:, i2[1]]), X[:, 3]); rtol = 1e-4, atol = 1e-5)
    end
    # deletion + compaction keep heads aligned with vectors
    db = KNNDatabase{T}(X; metric = m, k, filter = f, chunksize = 128)
    foreach(id -> delete!(db, id), 1:3:n)
    ref2 = KNNDatabase{T}(X; metric = m, k, chunksize = 128)
    foreach(id -> delete!(db, id), 1:3:n); foreach(id -> delete!(ref2, id), 1:3:n)
    Q = X[:, 2:5:200]
    a = knn(db, Q; k); b = knn(ref2, Q; k)
    @test all(isapprox.(a[2], b[2]; rtol = 1e-4, atol = 1e-5, nans = true))
    compact!(db)
    reset_stats!(db)
    c = knn(db, Q; k)
    @test all(isapprox.(c[2], b[2]; rtol = 1e-4, atol = 1e-5, nans = true))
    @test filter_stats(db).refined < filter_stats(db).scanned
    ids3, sc3 = stream_knn!(db, X[:, 1:100]; batchsize = 40)
    ids4, sc4 = stream_knn!(ref2, X[:, 1:100]; batchsize = 40)
    @test all(isapprox.(sc3, sc4; rtol = 1e-4, atol = 1e-5, nans = true))
end

@testset "HeadFilter argument checks" begin
    @test_throws ArgumentError HeadFilter(randn(8, 3))                   # not orthonormal
    @test_throws ArgumentError HeadFilter(ones(3, 8))                     # more head dims than dim
    B = Matrix(qr(randn(8, 3)).Q)
    @test_throws ArgumentError KNNDatabase(5; filter = HeadFilter(B))      # dim mismatch
    @test_throws ArgumentError KNNDatabase(5; filter = HeadFilter(6))
    @test_throws ArgumentError HeadFilter(6; coarse = 7)
    @test_throws ArgumentError HeadFilter(6; gemm = 7)
    @test_throws ArgumentError HeadFilter(6; gemm = 4, coarse = 5)
    @test HeadFilter(512).coarse == 0
    @test HeadFilter(1024; gemm = 512, coarse = 128).coarse == 128
    db = KNNDatabase(8; filter = HeadFilter(B))
    @test StreamingKNN.hasfilter(db)
    @test db.filter.basis isa Matrix{Float32}
    @test !StreamingKNN.hasfilter(KNNDatabase(8))
    X = randn(8, 50)
    @test 0 < head_energy(X, 3) < 1
    @test pca_basis(X, 8)' * pca_basis(X, 8) ≈ I atol = 1e-5
end

@testset "warm start: append! == one-by-one ($(T), $(typeof(m)), filter=$(usefilter))" for T in (Float16, Float32),
        m in METRICS, usefilter in (false, true)
    rng = MersenneTwister(hash((T, typeof(m), usefilter)))
    d, n, k = 24, 260, 6
    X = spectral_data(rng, d, n)
    # duplicates that straddle block boundaries, and a run of identical columns
    for (a, b) in ((70, 3), (71, 69), (140, 70), (141, 139), (210, 5), (211, 140))
        X[:, a] = X[:, b]
    end
    X[:, 100:103] .= X[:, 100]
    f = usefilter ? HeadFilter(pca_basis(X[:, 1:120], 5); coarse = 2, ncold = 8) : NoFilter()
    ref = KNNDatabase{T}(d; metric = m, k, filter = f, chunksize = 32)
    for t in 1:n
        push!(ref, X[:, t])
    end
    for block in (1, 7, 64, 1000)                       # 7 and 64 split the duplicate pairs
        db = KNNDatabase{T}(d; metric = m, k, filter = f, chunksize = 32)
        ids = append!(db, X; block)
        @test ids == 1:n
        @test nseen(db) == nseen(ref) == n
        @test length(db) == length(ref)
        @test nslots(db) == nslots(ref)
        @test [representative(db, t) for t in 1:n] == [representative(ref, t) for t in 1:n]
        @test [db[t] for t in 1:n] == [ref[t] for t in 1:n]
        @test duplicates(db, 3) == duplicates(ref, 3)
        @test duplicates(db, 100) == duplicates(ref, 100) == [100, 101, 102, 103]
        a = knn(db, X[:, 1:40]; k); b = knn(ref, X[:, 1:40]; k)
        @test a[1] == b[1]
        @test all(isapprox.(a[2], b[2]; rtol = 1e-4, atol = 1e-5, nans = true))
    end
    # warm start then go online: same answers as streaming the whole thing
    half = 130
    warm = KNNDatabase{T}(d; metric = m, k, filter = f, chunksize = 32)
    append!(warm, X[:, 1:half]; block = 16)
    wids, wsc = stream_knn!(warm, X[:, half+1:n]; batchsize = 17)
    full = KNNDatabase{T}(d; metric = m, k, filter = f, chunksize = 32)
    fids, fsc = stream_knn!(full, X; batchsize = 17)
    @test wids == fids[:, half+1:n]
    @test all(isapprox.(wsc, fsc[:, half+1:n]; rtol = 1e-4, atol = 1e-5, nans = true))
    @test_throws DimensionMismatch append!(warm, randn(d + 1, 3))
    @test_throws ArgumentError append!(warm, X[:, 1:3]; block = 0)
end

@testset "recall" begin
    truth = [1 2 3; 4 5 6; 7 8 9]
    @test recall(truth, truth) == 1.0
    @test recall(truth, [1 2 3; 4 5 6; 0 0 0]) ≈ 2 / 3
    @test recall(truth, [1 2 3; 4 5 6; 7 8 9]; k = 2) == 1.0
    @test recall(truth, [9 9 9; 9 9 9; 9 9 9]) ≈ 1 / 9
    @test recall([0 1; 0 0], [5 1; 6 7]) == 1.0          # query with no truth is skipped
    @test isnan(recall(zeros(Int, 2, 2), ones(Int, 2, 2)))
    @test_throws DimensionMismatch recall(truth, truth[:, 1:2])
    @test_throws ArgumentError recall(truth, truth[1:2, :]; k = 3)

    # score-tolerant variant
    ts = [0.9 0.9 0.9; 0.8 0.8 0.8; 0.7 0.7 0.7]
    pred = [10 2 3; 4 5 6; 7 8 9]                 # query 1 returns id 10 instead of 1
    ps = copy(ts)                                 # ... with an identical score (a tie)
    @test recall(truth, pred) ≈ 8 / 9
    @test recall(truth, ts, pred, ps) == 1.0      # ties are forgiven even with zero tolerance
    ps[1, 1] = 0.69                               # now id 10 is genuinely worse than the k-th true score
    @test recall(truth, ts, pred, ps) ≈ 8 / 9
    @test recall(truth, ts, pred, ps; atol = 0.02) == 1.0
    @test recall(truth, ts, pred, ps; rtol = 0.02) == 1.0
    @test recall(truth, ts, pred, ps; atol = 0.005) ≈ 8 / 9
    # Euclidean: smaller is better
    tsd = [0.1 0.1 0.1; 0.2 0.2 0.2; 0.3 0.3 0.3]
    psd = copy(tsd); psd[1, 1] = 0.31
    @test recall(truth, tsd, pred, psd; metric = Euclidean()) ≈ 8 / 9
    @test recall(truth, tsd, pred, psd; metric = Euclidean(), atol = 0.02) == 1.0
    @test recall(truth, tsd, pred, psd; metric = InnerProduct()) == 1.0        # 0.31 ≥ 0.3 counts as a hit when larger is better
    # a query cannot score above 1 even if many predictions clear the threshold
    @test recall([1 ; 0 ;; 2 ; 0], [0.5 0.5; 0.5 0.5], [7 2; 8 9], [0.9 0.9; 0.9 0.9]) == 1.0
    # duplicate predictions count once
    @test recall(truth, ts, [10 2 3; 10 5 6; 10 8 9], fill(0.95, 3, 3)) ≈ (1 / 3 + 1 + 1) / 3
    # smaller k uses the k-th true score
    @test recall(truth, ts, pred, ps; k = 2) ≈ (1 / 2 + 1 + 1) / 3            # 0.69 < 0.8, the 2nd true score
    @test recall(truth, ts, [10 2 3; 4 5 6; 7 8 9], [0.79 0.9 0.9; 0.8 0.8 0.8; 0.7 0.7 0.7]; k = 2) ≈ (1 / 2 + 1 + 1) / 3
    @test_throws DimensionMismatch recall(truth, ts[1:2, :], pred, ps)
    @test_throws ArgumentError recall(truth, ts, pred, ps; atol = -1)
end

@testset "Float64 storage computes in Float64" begin
    db = KNNDatabase{Float64}(4; metric = Euclidean())
    @test StreamingKNN.computetype(db) === Float64
    push!(db, [0.0, 0, 0, 0])
    _, sc = knn(db, [3.0, 4, 0, 0]; k = 1)
    @test sc[1] ≈ 5.0
end

end
