"""
    recall(truth, pred; k = size(pred, 1)) -> Float64

Mean recall@k of predicted neighbour ids against ground truth, both `k′ × n`
matrices as returned by [`knn`](@ref) or [`stream_knn!`](@ref), best first,
padded with `0`.  For every query the first `k` rows of each are compared;
queries with no true neighbour (e.g. the first element of a stream) are
skipped.  Ties are not accounted for: a prediction that returns a different
vector with an identical score counts as a miss.  See the score-tolerant
method below for that case.

    recall(truth_ids, truth_scores, pred_ids, pred_scores;
           k = size(pred_ids, 1), metric = InnerProduct(), atol = 0, rtol = 0) -> Float64

Score-tolerant recall@k.  A predicted neighbour is a hit if its id is among
the first `k` true ids **or** its score is at least as good as the k-th true
score up to a tolerance `atol + rtol * |k-th true score|` (larger scores are
better for `InnerProduct` and `Cosine`, smaller for `Euclidean`).  Each
predicted id counts at most once and a query contributes at most 1.
`pred_scores` must be the exact scores of the predicted vectors; an
approximate index that reports estimated scores should recompute them first.
With `atol = rtol = 0` the method only forgives exact ties.
"""
function recall(truth::AbstractMatrix{<:Integer}, pred::AbstractMatrix{<:Integer}; k::Integer = size(pred, 1))
    _check_recall_args(truth, pred, k)
    total = 0.0
    nq = 0
    seen = Set{Int}()
    @inbounds for q in axes(truth, 2)
        ntrue = _collect_truth!(seen, truth, q, k)
        ntrue == 0 && continue
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

function recall(truth::AbstractMatrix{<:Integer}, truth_scores::AbstractMatrix{<:Real},
                pred::AbstractMatrix{<:Integer}, pred_scores::AbstractMatrix{<:Real};
                k::Integer = size(pred, 1), metric::Metric = InnerProduct(), atol::Real = 0, rtol::Real = 0)
    _check_recall_args(truth, pred, k)
    size(truth_scores) == size(truth) || throw(DimensionMismatch("truth_scores must match truth"))
    size(pred_scores) == size(pred) || throw(DimensionMismatch("pred_scores must match pred"))
    atol >= 0 && rtol >= 0 || throw(ArgumentError("tolerances must be non-negative"))
    lib = larger_is_better(metric)
    total = 0.0
    nq = 0
    seen = Set{Int}()
    used = Set{Int}()
    @inbounds for q in axes(truth, 2)
        ntrue = _collect_truth!(seen, truth, q, k)
        ntrue == 0 && continue
        kth = Float64(truth_scores[ntrue, q])          # worst score among the true neighbours
        tol = atol + rtol * abs(kth)
        empty!(used)
        hits = 0
        for r in 1:k
            p = pred[r, q]
            p == 0 && break
            p in used && continue
            push!(used, p)
            s = Float64(pred_scores[r, q])
            good = lib ? s >= kth - tol : s <= kth + tol
            if p in seen || good
                hits += 1
                hits == ntrue && break
            end
        end
        total += hits / ntrue
        nq += 1
    end
    nq == 0 ? NaN : total / nq
end

function _check_recall_args(truth, pred, k)
    size(truth, 2) == size(pred, 2) || throw(DimensionMismatch("different number of queries"))
    k <= size(truth, 1) || throw(ArgumentError("ground truth has fewer than k = $k rows"))
    k <= size(pred, 1) || throw(ArgumentError("prediction has fewer than k = $k rows"))
end

"Fill `seen` with the first `k` true ids of query `q`; return how many there are."
@inline function _collect_truth!(seen::Set{Int}, truth, q, k)
    empty!(seen)
    @inbounds for r in 1:k
        t = truth[r, q]
        t == 0 && break
        push!(seen, t)
    end
    length(seen)
end
