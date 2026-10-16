"""
    recall(truth, pred; k = size(pred, 1)) -> Float64

Mean recall@k of predicted neighbour ids against ground truth, both `k′ × n`
matrices as returned by [`knn`](@ref) or [`stream_knn!`](@ref), best first,
padded with `0`.  For every query the first `k` rows of each are compared;
queries with no true neighbour (e.g. the first element of a stream) are
skipped.  Ties are not accounted for: an approximate method that returns a
different vector with an identical score is counted as a miss.
"""
function recall(truth::AbstractMatrix{<:Integer}, pred::AbstractMatrix{<:Integer}; k::Integer = size(pred, 1))
    size(truth, 2) == size(pred, 2) || throw(DimensionMismatch("different number of queries"))
    k <= size(truth, 1) || throw(ArgumentError("ground truth has fewer than k = $k rows"))
    k <= size(pred, 1) || throw(ArgumentError("prediction has fewer than k = $k rows"))
    total = 0.0
    nq = 0
    seen = Set{Int}()
    @inbounds for q in axes(truth, 2)
        empty!(seen)
        for r in 1:k
            t = truth[r, q]
            t == 0 && break
            push!(seen, t)
        end
        isempty(seen) && continue
        ntrue = length(seen)
        hits = 0
        for r in 1:k
            p = pred[r, q]
            p == 0 && break
            if p in seen                  # each true neighbour counts once
                delete!(seen, p)
                hits += 1
            end
        end
        total += hits / ntrue
        nq += 1
    end
    nq == 0 ? NaN : total / nq
end
