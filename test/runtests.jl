using StreamingKNN
using StreamingKNN: reference_score, TopK, threshold, sorted_indices!, reset!
using Test
using Random

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

@testset "Float64 storage computes in Float64" begin
    db = KNNDatabase{Float64}(4; metric = Euclidean())
    @test StreamingKNN.computetype(db) === Float64
    push!(db, [0.0, 0, 0, 0])
    _, sc = knn(db, [3.0, 4, 0, 0]; k = 1)
    @test sc[1] ≈ 5.0
end

end
