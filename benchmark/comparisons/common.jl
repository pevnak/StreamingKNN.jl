using Serialization, LinearAlgebra, Printf, Statistics
"Data file: a serialized `Matrix{Float16}`, one vector per column.  Override with STREAMINGKNN_DATA."
const DATA = get(ENV, "STREAMINGKNN_DATA",
                 normpath(joinpath(@__DIR__, "..", "..", "bench", "knntest_1.jls")))
load_prefix(n) = (X = deserialize(DATA); Matrix(view(X, :, 1:n)))

"Mean recall@k of `pred` ids against causal ground-truth ids (0-padded), skipping empty queries."
function recall_at(truth::Matrix{Int}, pred::Matrix{Int}, k::Int)
    tot = 0.0; nq = 0
    for q in axes(truth, 2)
        s = Set(t for t in view(truth, 1:k, q) if t != 0)
        isempty(s) && continue
        hits = count(p -> p in s, view(pred, 1:k, q))
        tot += hits / length(s); nq += 1
    end
    nq == 0 ? NaN : tot / nq
end

"""
    canonical_map(X) -> Vector{Int}

`c[i]` is the first column index holding a vector bitwise identical to column `i`.
Needed to compare libraries fairly: StreamingKNN folds duplicates and reports the
first occurrence, other libraries report whichever copy their index happens to find.
"""
function canonical_map(X::AbstractMatrix)
    seen = Dict{UInt64,Vector{Int}}()
    c = collect(1:size(X, 2))
    for j in axes(X, 2)
        col = view(X, :, j)
        h = hash(col)
        bucket = get!(Vector{Int}, seen, h)
        found = 0
        for i in bucket
            view(X, :, i) == col && (found = i; break)
        end
        if found == 0
            push!(bucket, j)
        else
            c[j] = c[found]
        end
    end
    c
end

canonicalise(ids::Matrix{Int}, c::Vector{Int}) = [v == 0 ? 0 : c[v] for v in ids]
