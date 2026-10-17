"""
    Filter

Opt-in exact pruning for the search kernel.  See [`HeadFilter`](@ref).
"""
abstract type Filter end

"No pruning: every stored vector is scored with a full inner product."
struct NoFilter <: Filter end

"""
    HeadFilter(ndims; slack = 1e-3)
    HeadFilter(basis::AbstractMatrix; slack = 1e-3)

Exact filter-and-refine search.  Every vector `x` is split into a *head*
`h = Bᵀx` of `ndims` coordinates and a tail.  The head is stored separately
and the matrix products of the search run on heads only, so they cost
`ndims / dim` of a full scan.  The Cauchy–Schwarz bound

    ⟨q, x⟩ ≤ ⟨Bᵀq, h⟩ + ‖q_tail‖ ‖x_tail‖

is valid for every pair, so a vector whose bound cannot beat the current
k-th best score is skipped, and only the survivors get an exact score from
the full stored vector.  Results are identical to the unfiltered search;
only the speed depends on how much energy the heads capture.

* `HeadFilter(basis; gemm)` uses an orthonormal `dim × ndims` matrix,
  typically from [`pca_basis`](@ref).  The matrix product runs on the leading
  `gemm` coordinates (default: all of them).  When `gemm < ndims` the
  survivors of that bound are bounded again with the remaining stored head
  coordinates before the full vector is touched, which is far cheaper than a
  full refinement.  On 4096-dimensional text embeddings
  `HeadFilter(pca_basis(X, 1024); gemm = 512)` is a good configuration.
* `HeadFilter(ndims; gemm)` uses the leading coordinates without any
  rotation, for Matryoshka-style embeddings whose energy is concentrated in
  the first coordinates.  Verify this with [`head_energy`](@ref) first; on
  embeddings without that property it prunes little.

`slack` is a relative safety margin added to the bound as
`slack · ‖q‖ ‖x‖`; it covers the Float32 rounding of the matrix products and
of the stored heads.  `1e-3` is ample and costs almost no pruning power.

`coarse > 0` enables an optional *threshold pass* that bounds every
candidate with only the leading `coarse` head coordinates, keeps the `ncold`
best per query and scores them exactly, so that every query starts the main
pass with a threshold close to its final k-th score.  It halves the number
of candidates that reach the second stage, but on the benchmark data its own
matrix product and bookkeeping cost more than that saves, so it is off by
default.
"""
struct HeadFilter{C<:AbstractFloat} <: Filter
    ndims::Int           # stored head coordinates
    gemm::Int            # leading head coordinates scored by the matrix product (stage 1)
    coarse::Int          # leading head coordinates used by the threshold pass (0 disables it)
    ncold::Int           # candidates per query refined exactly by the threshold pass
    basis::Union{Nothing,Matrix{C}}
    slack::C
end

function HeadFilter(ndims::Integer; gemm::Integer = ndims, coarse::Integer = default_coarse(gemm),
                    ncold::Integer = 128, slack::Real = 1e-3)
    check_stages(ndims, gemm, coarse)
    HeadFilter{Float32}(Int(ndims), Int(gemm), Int(coarse), Int(ncold), nothing, Float32(slack))
end

function HeadFilter(basis::AbstractMatrix{<:Real}; gemm::Integer = size(basis, 2),
                    coarse::Integer = default_coarse(gemm), ncold::Integer = 128, slack::Real = 1e-3)
    C = eltype(basis) <: AbstractFloat ? eltype(basis) : Float32
    B = Matrix{C}(basis)
    check_orthonormal(B)
    check_stages(size(B, 2), gemm, coarse)
    HeadFilter{C}(size(B, 2), Int(gemm), Int(coarse), Int(ncold), B, C(slack))
end

default_coarse(gemm) = 0
function check_stages(ndims, gemm, coarse)
    1 <= gemm <= ndims || throw(ArgumentError("gemm must be in 1:$ndims"))
    0 <= coarse <= gemm || throw(ArgumentError("coarse must be in 0:$gemm"))
    nothing
end

"Whether the filter has a second bounding stage on head coordinates `gemm+1:ndims`."
has_stage2(f::HeadFilter) = f.ndims > f.gemm

function check_orthonormal(B::AbstractMatrix)
    size(B, 2) <= size(B, 1) || throw(ArgumentError("basis must be dim × ndims with ndims ≤ dim"))
    G = B' * B
    err = maximum(abs, G - I)
    err < 1e-3 || throw(ArgumentError("basis columns are not orthonormal (max deviation $err)"))
    nothing
end

"Convert the filter to the compute type `C` of a database and check its dimension."
adapt_filter(f::NoFilter, ::Type, dim::Int) = f
function adapt_filter(f::HeadFilter, ::Type{C}, dim::Int) where {C}
    1 <= f.ndims <= dim || throw(ArgumentError("filter has $(f.ndims) head coordinates, vectors have $dim"))
    if f.basis === nothing
        HeadFilter{C}(f.ndims, f.gemm, f.coarse, f.ncold, nothing, C(f.slack))
    else
        size(f.basis, 1) == dim || throw(ArgumentError("basis has $(size(f.basis, 1)) rows, vectors have $dim"))
        HeadFilter{C}(f.ndims, f.gemm, f.coarse, f.ncold, Matrix{C}(f.basis), C(f.slack))
    end
end

hasfilter(::NoFilter) = false
hasfilter(::HeadFilter) = true
head_dims(::NoFilter) = 0
head_dims(f::HeadFilter) = f.ndims

"""
    heads!(H, filter, X)

Write the heads of the columns of `X` (`dim × n`, compute type) into `H`
(`ndims × n`).
"""
function heads!(H::AbstractMatrix{C}, f::HeadFilter{C}, X::AbstractMatrix{C}) where {C}
    if f.basis === nothing
        copyto!(H, view(X, 1:f.ndims, :))
    else
        mul!(H, transpose(f.basis), X)
    end
    H
end

"""
    pca_basis(X, ndims; nsample = 50_000) -> Matrix{Float32}

Orthonormal `dim × ndims` basis of the top principal directions of the
columns of `X` (uncentered, i.e. the directions that capture the most
squared norm, which is what the [`HeadFilter`](@ref) bound needs).  At most
`nsample` evenly spaced columns are used.  For a streaming database learn it
on the initial content or on the first batch; a basis that fits the data
less well only reduces pruning, never correctness.
"""
function pca_basis(X::AbstractMatrix, ndims::Integer; nsample::Integer = 50_000)
    d, n = size(X)
    1 <= ndims <= d || throw(ArgumentError("ndims must be in 1:$d"))
    cols = n <= nsample ? (1:n) : unique(round.(Int, range(1, n; length = nsample)))
    S = Matrix{Float32}(undef, d, length(cols))
    for (j, c) in enumerate(cols)
        @inbounds for i in 1:d
            S[i, j] = Float32(X[i, c])
        end
    end
    G = Symmetric(Matrix{Float64}(S * S'))
    E = eigen(G, d - ndims + 1:d)              # the ndims largest eigenpairs, ascending
    Matrix{Float32}(reverse(E.vectors; dims = 2))
end

"""
    head_energy(X, ndims; basis = nothing, nsample = 10_000) -> Float64

Fraction of the squared norm of the columns of `X` captured by the head,
either the leading `ndims` coordinates or the projection onto `basis`.
Values close to 1 mean strong pruning; `ndims / dim` means none.
"""
function head_energy(X::AbstractMatrix, ndims::Integer; basis = nothing, nsample::Integer = 10_000)
    d, n = size(X)
    cols = n <= nsample ? (1:n) : unique(round.(Int, range(1, n; length = nsample)))
    total = 0.0
    head = 0.0
    g = Vector{Float64}(undef, basis === nothing ? 0 : size(basis, 2))
    for c in cols
        x = Float64.(view(X, :, c))
        total += sum(abs2, x)
        if basis === nothing
            head += sum(abs2, view(x, 1:ndims))
        else
            mul!(g, transpose(basis), x)
            head += sum(abs2, g)
        end
    end
    head / total
end
