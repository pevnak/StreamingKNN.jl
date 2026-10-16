"""
    ChunkedStorage{T}(dim; chunksize)

Column-major storage of `dim`-dimensional vectors in fixed-width chunks.
Appending never reallocates or copies previously stored vectors, and the
chunk is the unit of work of the search kernel.
"""
mutable struct ChunkedStorage{T}
    dim::Int
    chunksize::Int
    chunks::Vector{Matrix{T}}
    n::Int
end

ChunkedStorage{T}(dim::Integer; chunksize::Integer = 1 << 15) where {T} =
    ChunkedStorage{T}(Int(dim), Int(chunksize), Matrix{T}[], 0)

Base.length(s::ChunkedStorage) = s.n
nchunks(s::ChunkedStorage) = length(s.chunks)

"Chunk index and column of slot `i`."
@inline chunk_of(s::ChunkedStorage, i::Integer) =
    ((i - 1) ÷ s.chunksize + 1, (i - 1) % s.chunksize + 1)

"Number of filled columns in chunk `c`."
@inline nfilled(s::ChunkedStorage, c::Integer) =
    clamp(s.n - (c - 1) * s.chunksize, 0, s.chunksize)

"First slot index stored in chunk `c`."
@inline chunk_offset(s::ChunkedStorage, c::Integer) = (c - 1) * s.chunksize

@inline function Base.getindex(s::ChunkedStorage, i::Integer)
    @boundscheck 1 <= i <= s.n || throw(BoundsError(s, i))
    c, j = chunk_of(s, i)
    @inbounds view(s.chunks[c], :, j)
end

"Append a vector (already of element type `T`) and return its slot."
function Base.push!(s::ChunkedStorage{T}, x::AbstractVector) where {T}
    length(x) == s.dim || throw(DimensionMismatch("expected length $(s.dim), got $(length(x))"))
    c, j = chunk_of(s, s.n + 1)
    if c > length(s.chunks)
        push!(s.chunks, Matrix{T}(undef, s.dim, s.chunksize))
    end
    @inbounds copyto!(view(s.chunks[c], :, j), x)
    s.n += 1
    return s.n
end

"Copy slot `src` onto slot `dst` (used by compaction)."
function movecolumn!(s::ChunkedStorage, dst::Integer, src::Integer)
    dst == src && return s
    cd, jd = chunk_of(s, dst)
    cs, js = chunk_of(s, src)
    @inbounds copyto!(view(s.chunks[cd], :, jd), view(s.chunks[cs], :, js))
    s
end

"Shrink to `n` slots, releasing chunks that became empty."
function truncate!(s::ChunkedStorage, n::Integer)
    s.n = Int(n)
    needed = n == 0 ? 0 : chunk_of(s, n)[1]
    resize!(s.chunks, needed)
    s
end
