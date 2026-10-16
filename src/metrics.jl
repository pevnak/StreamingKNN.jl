"""
    Metric

Abstract supertype of the similarity measures understood by the database.

All three metrics reduce to a single matrix product `S = Xᵀ Q` between the
stored vectors and the query batch.  Every stored vector `j` carries two
coefficients `scale[j]` and `shift[j]` such that

    key = scale[j] * S[j, b] + shift[j]

is a *ranking key* for which larger is better, independently of the metric:

| metric         | key                       | reported score                 |
|----------------|---------------------------|--------------------------------|
| `InnerProduct` | `S`                       | `key`                          |
| `Cosine`       | `S / ‖x‖`                 | `key / ‖q‖`                    |
| `Euclidean`    | `2S - ‖x‖²`               | `sqrt(‖q‖² - key)`             |

Deleted vectors get `scale = 0`, `shift = -Inf`, so their key is `-Inf` and
they are never selected without any extra branch in the hot loop.
"""
abstract type Metric end

"Maximum inner product search.  Score reported: `⟨x, q⟩`."
struct InnerProduct <: Metric end

"Cosine similarity.  Score reported: `⟨x, q⟩ / (‖x‖ ‖q‖)`; zero vectors score 0."
struct Cosine <: Metric end

"Euclidean distance.  Score reported: `‖x - q‖` (smaller is better; results are sorted ascending)."
struct Euclidean <: Metric end

"""
    slot_coeffs(metric, n2) -> (scale, shift)

Per-vector coefficients from its squared norm `n2` (see [`Metric`](@ref)).
"""
@inline slot_coeffs(::InnerProduct, n2::C) where {C} = (one(C), zero(C))
@inline slot_coeffs(::Cosine, n2::C) where {C} = (n2 > 0 ? inv(sqrt(n2)) : zero(C), zero(C))
@inline slot_coeffs(::Euclidean, n2::C) where {C} = (C(2), -n2)

"Coefficients that make a slot invisible to every query."
@inline dead_coeffs(::Type{C}) where {C} = (zero(C), typemin(C))

"""
    finalscore(metric, key, qn2)

Turn a ranking key back into the user-facing score, given the squared norm
`qn2` of the query.
"""
@inline finalscore(::InnerProduct, key, qn2) = key
@inline finalscore(::Cosine, key, qn2) = qn2 > 0 ? key / sqrt(qn2) : zero(key)
@inline finalscore(::Euclidean, key, qn2) = sqrt(max(zero(key), qn2 - key))

"""
    reference_score(metric, x, q)

Straightforward, unoptimised score between two vectors.  Used by the tests
and handy for sanity checks; not used by the search kernel.
"""
function reference_score(::InnerProduct, x, q)
    P = promote_type(Float32, eltype(x), eltype(q))
    s = zero(P)
    @inbounds for i in eachindex(x, q)
        s += P(x[i]) * P(q[i])
    end
    s
end
function reference_score(::Cosine, x, q)
    s = reference_score(InnerProduct(), x, q)
    nx = sqrt(reference_score(InnerProduct(), x, x))
    nq = sqrt(reference_score(InnerProduct(), q, q))
    (nx > 0 && nq > 0) ? s / (nx * nq) : zero(s)
end
function reference_score(::Euclidean, x, q)
    P = promote_type(Float32, eltype(x), eltype(q))
    s = zero(P)
    @inbounds for i in eachindex(x, q)
        d = P(x[i]) - P(q[i])
        s += d * d
    end
    sqrt(s)
end
