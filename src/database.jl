"Counters of the filtered search, see [`HeadFilter`](@ref) and [`filter_stats`](@ref)."
mutable struct FilterStats
    scanned::Int      # candidate (slot, query) pairs bounded
    stage2::Int       # pairs that passed the first bound and were bounded again with the rest of the head
    refined::Int      # pairs that needed an exact score
end

"""
    KNNDatabase{T,C,M,F}

Exact nearest-neighbour database for streaming vectors.

* `T` – storage element type (default `Float16`).
* `C` – compute type used for the matrix products (`Float32` unless `T == Float64`).
* `M` – the [`Metric`](@ref).

Terminology
-----------
* **id** – position in the stream, assigned by every insertion (duplicates included).
  Ids are stable for the lifetime of the database.
* **slot** – position of a unique vector inside the storage.  Several ids can
  share a slot when the vectors are `isequal` after conversion to `T`.
  Search results report the smallest alive id of each slot.
"""
mutable struct KNNDatabase{T,C<:AbstractFloat,M<:Metric,F<:Filter}
    metric::M
    k::Int
    storage::ChunkedStorage{T}
    direct_batch::Int                 # use the direct dot kernel for batches up to this size
    # filter ----------------------------------------------------------------
    filter::F
    head::ChunkedStorage{T}           # ndims × n heads (empty without filter)
    tail_c::Vector{C}                 # per slot: ‖x‖ beyond the coarse sub-head
    tail_g::Vector{C}                 # per slot: ‖x‖ beyond the gemm sub-head
    tail_h::Vector{C}                 # per slot: ‖x‖ beyond the stored head
    headerr::Vector{C}                # per slot: head rounding error + slack·‖x‖
    stats::FilterStats
    # per slot --------------------------------------------------------------
    scale::Vector{C}
    shift::Vector{C}
    norms2::Vector{C}
    rep_id::Vector{Int}               # smallest alive id of the slot, 0 when dead
    alive::BitVector
    nalive::Int
    # ids ------------------------------------------------------------------
    nseen::Int
    id2slot::Vector{Int}              # 0 when the id was deleted
    dups::Dict{Int,Vector{Int}}       # slot -> all alive ids (only when ≥ 2)
    # dedup ----------------------------------------------------------------
    hash2slot::Dict{UInt64,Int}
    collisions::Dict{UInt64,Vector{Int}}   # extra slots sharing a hash (practically never used)
    # scratch --------------------------------------------------------------
    ws::Any                           # Workspace{C} or nothing
end

default_compute_type(::Type{T}) where {T} = T === Float64 ? Float64 : Float32

"""
    KNNDatabase{T}(dim; metric = InnerProduct(), k = 10, filter = NoFilter(),
                   chunksize = 2^15, compute = ..., direct_batch = 16)
    KNNDatabase(X::AbstractMatrix; kwargs...)

Create an empty database for `dim`-dimensional vectors stored as `T`, or one
initialised with the columns of `X` (stored as `T = eltype(X)` unless given).
Pass `filter = HeadFilter(...)` for exact filter-and-refine search, see
[`HeadFilter`](@ref).
"""
function KNNDatabase{T}(dim::Integer;
                        metric::Metric = InnerProduct(),
                        k::Integer = 10,
                        filter::Filter = NoFilter(),
                        chunksize::Integer = 1 << 15,
                        compute::Type{C} = default_compute_type(T),
                        direct_batch::Integer = 16) where {T,C<:AbstractFloat}
    k >= 1 || throw(ArgumentError("k must be positive"))
    f = adapt_filter(filter, C, Int(dim))
    KNNDatabase{T,C,typeof(metric),typeof(f)}(
        metric, Int(k), ChunkedStorage{T}(dim; chunksize), Int(direct_batch),
        f, ChunkedStorage{T}(head_dims(f); chunksize), C[], C[], C[], C[], FilterStats(0, 0, 0),
        C[], C[], C[], Int[], BitVector(), 0,
        0, Int[], Dict{Int,Vector{Int}}(),
        Dict{UInt64,Int}(), Dict{UInt64,Vector{Int}}(),
        nothing)
end

KNNDatabase(dim::Integer; kwargs...) = KNNDatabase{Float16}(dim; kwargs...)

function KNNDatabase{T}(X::AbstractMatrix; kwargs...) where {T}
    db = KNNDatabase{T}(size(X, 1); kwargs...)
    append!(db, X)
    db
end
KNNDatabase(X::AbstractMatrix; kwargs...) = KNNDatabase{eltype(X)}(X; kwargs...)

# ---------------------------------------------------------------------------
# accessors

storagetype(::KNNDatabase{T}) where {T} = T
computetype(::KNNDatabase{T,C}) where {T,C} = C
metric(db::KNNDatabase) = db.metric
dimension(db::KNNDatabase) = db.storage.dim
hasfilter(db::KNNDatabase) = hasfilter(db.filter)

"""
    filter_stats(db) -> (scanned, stage2, refined, fraction)

Counters of the filtered search since the last [`reset_stats!`](@ref):
candidate pairs bounded, pairs bounded a second time with the rest of the
head, pairs that needed an exact score, and the ratio refined / scanned.
"""
function filter_stats(db::KNNDatabase)
    s = db.stats
    (scanned = s.scanned, stage2 = s.stage2, refined = s.refined,
     fraction = s.scanned == 0 ? NaN : s.refined / s.scanned)
end
reset_stats!(db::KNNDatabase) = (db.stats.scanned = 0; db.stats.stage2 = 0; db.stats.refined = 0; db)

"Number of alive unique vectors."
Base.length(db::KNNDatabase) = db.nalive
"Number of ids assigned so far (every insertion, duplicates included)."
nseen(db::KNNDatabase) = db.nseen
"Number of slots in storage, dead ones included (shrinks after [`compact!`](@ref))."
nslots(db::KNNDatabase) = db.storage.n

isdeleted(db::KNNDatabase, id::Integer) = slot_of(db, id) == 0

@inline function slot_of(db::KNNDatabase, id::Integer)
    1 <= id <= db.nseen || throw(BoundsError(db.id2slot, id))
    @inbounds db.id2slot[id]
end

"Stored vector of `id` (a view of element type `T`)."
function Base.getindex(db::KNNDatabase, id::Integer)
    s = slot_of(db, id)
    s == 0 && throw(KeyError(id))
    db.storage[s]
end

"""
    duplicates(db, id) -> Vector{Int}

All alive ids whose vector is identical to the vector of `id` (including `id`).
"""
function duplicates(db::KNNDatabase, id::Integer)
    s = slot_of(db, id)
    s == 0 && throw(KeyError(id))
    v = get(db.dups, s, nothing)
    v === nothing ? [db.rep_id[s]] : copy(v)
end

"Representative (smallest alive) id of the slot that `id` belongs to."
function representative(db::KNNDatabase, id::Integer)
    s = slot_of(db, id)
    s == 0 && throw(KeyError(id))
    db.rep_id[s]
end

function Base.show(io::IO, db::KNNDatabase{T,C,M}) where {T,C,M}
    print(io, "KNNDatabase{$T,$C}($(M()), dim=$(dimension(db)), k=$(db.k), ",
          "unique=$(length(db)), ids=$(nseen(db)), slots=$(nslots(db))")
    hasfilter(db) && print(io, ", filter=HeadFilter($(db.filter.ndims), gemm=$(db.filter.gemm))")
    print(io, ")")
end

# ---------------------------------------------------------------------------
# hashing & dedup

"""
    vhash(x)

Hash of the bit patterns of a vector.  Two vectors are duplicates when they
are bitwise identical after conversion to the storage type (so `0.0` and
`-0.0` are distinct, and equal `NaN` payloads are duplicates).  For the
common element types the bits are packed into 64-bit words before hashing.
"""
function vhash(x::AbstractVector{T}) where {T}
    h = hash(length(x))
    n = length(x)
    if T === Float16 || T === Int16 || T === UInt16
        i = firstindex(x)
        @inbounds while i + 3 <= lastindex(x)
            w = UInt64(bits16(x[i])) | UInt64(bits16(x[i+1])) << 16 |
                UInt64(bits16(x[i+2])) << 32 | UInt64(bits16(x[i+3])) << 48
            h = hash(w, h)
            i += 4
        end
        @inbounds while i <= lastindex(x)
            h = hash(UInt64(bits16(x[i])), h)
            i += 1
        end
    elseif T === Float32 || T === Int32 || T === UInt32
        i = firstindex(x)
        @inbounds while i + 1 <= lastindex(x)
            h = hash(UInt64(bits32(x[i])) | UInt64(bits32(x[i+1])) << 32, h)
            i += 2
        end
        @inbounds while i <= lastindex(x)
            h = hash(UInt64(bits32(x[i])), h)
            i += 1
        end
    elseif T === Float64 || T === Int64 || T === UInt64
        @inbounds for i in eachindex(x)
            h = hash(reinterpret(UInt64, x[i]), h)
        end
    else
        @inbounds for i in eachindex(x)
            h = hash(x[i], h)
        end
    end
    h
end
@inline bits16(v) = reinterpret(UInt16, v)
@inline bits32(v) = reinterpret(UInt32, v)

"Bitwise equality of two vectors (see [`vhash`](@ref))."
function vequal(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    length(a) == length(b) || return false
    if isbitstype(T)
        @inbounds for i in eachindex(a, b)
            a[i] === b[i] || return false                 # bitwise identity
        end
    else
        @inbounds for i in eachindex(a, b)
            isequal(a[i], b[i]) || return false
        end
    end
    true
end
vequal(a::AbstractVector, b::AbstractVector) = vequal(promote(a, b)...)

"Slot holding a vector `isequal` to `x`, or 0."
function find_slot(db::KNNDatabase, h::UInt64, x::AbstractVector)
    s = get(db.hash2slot, h, 0)
    s == 0 && return 0
    vequal(db.storage[s], x) && return s
    others = get(db.collisions, h, nothing)
    others === nothing && return 0
    for t in others
        vequal(db.storage[t], x) && return t
    end
    0
end

function register_hash!(db::KNNDatabase, h::UInt64, s::Int)
    if haskey(db.hash2slot, h)
        push!(get!(Vector{Int}, db.collisions, h), s)
    else
        db.hash2slot[h] = s
    end
end

function unregister_hash!(db::KNNDatabase, h::UInt64, s::Int)
    others = get(db.collisions, h, nothing)
    if db.hash2slot[h] == s
        if others === nothing
            delete!(db.hash2slot, h)
        else
            db.hash2slot[h] = popfirst!(others)
            isempty(others) && delete!(db.collisions, h)
        end
    else
        deleteat!(others, findfirst(==(s), others))
        isempty(others) && delete!(db.collisions, h)
    end
end

# ---------------------------------------------------------------------------
# insertion

function squared_norm(::Type{C}, x::AbstractVector) where {C}
    s = zero(C)
    @inbounds @simd for i in eachindex(x)
        v = C(x[i])
        s += v * v
    end
    s
end

"""
    insert_vector!(db, xt) -> (slot, isnew)

Assign a new id to `xt` (already converted to the storage type), given its
hash `h` and squared norm `n2`.  Returns the slot the id was attached to and
whether that slot was created by this call.
"""
function insert_vector!(db::KNNDatabase{T,C}, xt::AbstractVector{T}, h::UInt64, n2::C) where {T,C}
    id = db.nseen += 1
    s = find_slot(db, h, xt)
    if s != 0
        push!(db.id2slot, s)
        v = get!(db.dups, s) do
            [db.rep_id[s]]
        end
        push!(v, id)
        return s, false
    end
    s = push!(db.storage, xt)
    sc, sh = slot_coeffs(db.metric, n2)
    push!(db.scale, sc)
    push!(db.shift, sh)
    push!(db.norms2, n2)
    push!(db.rep_id, id)
    push!(db.alive, true)
    db.nalive += 1
    push!(db.id2slot, s)
    register_hash!(db, h, s)
    return s, true
end

"""
Columns per block of a bulk insertion.  Caps the temporary memory of
[`append!`](@ref) at roughly `BULK_BLOCK * (dim + ndims) * 4` bytes however
much is inserted, instead of growing with the whole input.
"""
const BULK_BLOCK = 1 << 14

"Below this many columns the threaded prepass costs more than it saves."
const PARALLEL_INSERT_MIN = 64

"The columns of `Xb` in the storage type, without copying when they already are."
staging(::Type{T}, Xb::AbstractMatrix{T}) where {T} = Xb
function staging(::Type{T}, Xb::AbstractMatrix) where {T}
    S = Matrix{T}(undef, size(Xb)...)
    if size(Xb, 2) >= PARALLEL_INSERT_MIN
        Threads.@threads :static for i in axes(Xb, 2)
            @inbounds copyto!(view(S, :, i), view(Xb, :, i))
        end
    else
        copyto!(S, Xb)
    end
    S
end

"Hashes and squared norms of every column, threaded for large blocks."
function insert_prepass!(hashes::Vector{UInt64}, norms::Vector{C}, S::AbstractMatrix, B::Int) where {C}
    if B >= PARALLEL_INSERT_MIN && Threads.nthreads() > 1
        Threads.@threads :static for i in 1:B
            @inbounds col = view(S, :, i)
            @inbounds hashes[i] = vhash(col)
            @inbounds norms[i] = squared_norm(C, col)
        end
    else
        @inbounds for i in 1:B
            col = view(S, :, i)
            hashes[i] = vhash(col)
            norms[i] = squared_norm(C, col)
        end
    end
    nothing
end

"""
    insert_batch!(db, Xb) -> newpos

Assign ids to all columns of `Xb` (dedup included) and return the positions
of the columns that created new slots, in order.  Hashes and squared norms
are computed in a threaded prepass; the heads of the new slots are computed
with one matrix product.
"""
function insert_batch!(db::KNNDatabase{T,C}, Xb::AbstractMatrix) where {T,C}
    size(Xb, 1) == dimension(db) || throw(DimensionMismatch("expected $(dimension(db)) rows, got $(size(Xb, 1))"))
    B = size(Xb, 2)
    B == 0 && return Int[]
    n0 = nslots(db)
    S = staging(T, Xb)
    hashes = Vector{UInt64}(undef, B)
    norms = Vector{C}(undef, B)
    insert_prepass!(hashes, norms, S, B)
    newpos = Int[]
    @inbounds for i in 1:B
        _, isnew = insert_vector!(db, view(S, :, i), hashes[i], norms[i])
        isnew && push!(newpos, i)
    end
    add_heads!(db, n0, length(newpos))
    newpos
end

add_heads!(db::KNNDatabase{T,C,M,NoFilter}, n0::Int, nnew::Int) where {T,C,M} = db

"Compute and store heads, tail norms and head errors of slots `n0+1 : n0+nnew`."
function add_heads!(db::KNNDatabase{T,C,M,<:HeadFilter}, n0::Int, nnew::Int) where {T,C,M}
    nnew == 0 && return db
    f = db.filter
    st = db.storage
    Xn = Matrix{C}(undef, st.dim, nnew)
    for j in 1:nnew
        @inbounds copyto!(view(Xn, :, j), st[n0 + j])
    end
    H = heads!(Matrix{C}(undef, f.ndims, nnew), f, Xn)
    ht = Vector{T}(undef, f.ndims)
    @inbounds for j in 1:nnew
        h = view(H, :, j)
        ht .= h                                      # stored (rounded) head
        rounding = zero(C)
        hn2 = zero(C)
        hn2c = zero(C)
        hn2g = zero(C)
        for i in 1:f.ndims
            rounding += abs2(h[i] - C(ht[i]))
            hn2 += abs2(h[i])
            i == f.coarse && (hn2c = hn2)
            i == f.gemm && (hn2g = hn2)
        end
        n2 = db.norms2[n0 + j]
        push!(db.head, ht)
        push!(db.tail_c, sqrt(max(zero(C), n2 - hn2c)))
        push!(db.tail_g, sqrt(max(zero(C), n2 - hn2g)))
        push!(db.tail_h, sqrt(max(zero(C), n2 - hn2)))
        push!(db.headerr, sqrt(rounding) + f.slack * sqrt(n2))
    end
    db
end

"""
    push!(db, x) -> id

Insert one vector without searching.  Returns the id assigned to it.
"""
function Base.push!(db::KNNDatabase, x::AbstractVector)
    length(x) == dimension(db) || throw(DimensionMismatch("expected length $(dimension(db)), got $(length(x))"))
    insert_batch!(db, reshape(x, :, 1))
    db.nseen
end

"""
    append!(db, X; block = BULK_BLOCK) -> ids

Insert the columns of `X` without searching, and return the assigned ids.
This is the **warm start** path: it is threaded, it never materialises more
than `block` columns of scratch space, and it skips the search entirely, so
loading a large initial set costs a few microseconds per vector instead of
the quadratic cost of streaming it.  The resulting database is identical to
one built by inserting the same vectors one at a time.
"""
function Base.append!(db::KNNDatabase, X::AbstractMatrix; block::Integer = BULK_BLOCK)
    size(X, 1) == dimension(db) || throw(DimensionMismatch("expected $(dimension(db)) rows, got $(size(X, 1))"))
    block >= 1 || throw(ArgumentError("block must be positive"))
    n = size(X, 2)
    first = db.nseen + 1
    sizehint!(db.id2slot, db.nseen + n)
    for lo in 1:block:n
        insert_batch!(db, view(X, :, lo:min(lo + block - 1, n)))
    end
    first:db.nseen
end

# ---------------------------------------------------------------------------
# deletion & compaction

function kill_slot!(db::KNNDatabase{T,C}, s::Int) where {T,C}
    unregister_hash!(db, vhash(db.storage[s]), s)
    db.alive[s] = false
    db.nalive -= 1
    db.scale[s], db.shift[s] = dead_coeffs(C)
    db.rep_id[s] = 0
    db
end

"""
    delete!(db, id)

Remove one id.  The underlying vector disappears from search results once all
ids sharing it are deleted.  Storage is reclaimed by [`compact!`](@ref).
"""
function Base.delete!(db::KNNDatabase, id::Integer)
    s = slot_of(db, id)
    s == 0 && return db
    db.id2slot[id] = 0
    v = get(db.dups, s, nothing)
    if v === nothing
        kill_slot!(db, s)
    else
        deleteat!(v, findfirst(==(id), v))
        db.rep_id[s] = v[1]
        length(v) == 1 && delete!(db.dups, s)
    end
    db
end

"Fraction of slots that are dead."
deadfraction(db::KNNDatabase) = nslots(db) == 0 ? 0.0 : 1 - db.nalive / nslots(db)

"""
    compact!(db)

Physically remove dead slots, moving alive vectors forward in place.  Ids are
unaffected; slots are renumbered.
"""
function compact!(db::KNNDatabase)
    n = nslots(db)
    db.nalive == n && return db
    remap = zeros(Int, n)
    w = 0
    for r in 1:n
        db.alive[r] || continue
        w += 1
        remap[r] = w
        w == r && continue
        movecolumn!(db.storage, w, r)
        if hasfilter(db)
            movecolumn!(db.head, w, r)
            db.tail_c[w] = db.tail_c[r]
            db.tail_g[w] = db.tail_g[r]
            db.tail_h[w] = db.tail_h[r]
            db.headerr[w] = db.headerr[r]
        end
        db.scale[w] = db.scale[r]
        db.shift[w] = db.shift[r]
        db.norms2[w] = db.norms2[r]
        db.rep_id[w] = db.rep_id[r]
    end
    truncate!(db.storage, w)
    if hasfilter(db)
        truncate!(db.head, w)
        resize!(db.tail_c, w); resize!(db.tail_g, w); resize!(db.tail_h, w); resize!(db.headerr, w)
    end
    resize!(db.scale, w); resize!(db.shift, w); resize!(db.norms2, w); resize!(db.rep_id, w)
    db.alive = trues(w)
    for id in 1:db.nseen
        s = db.id2slot[id]
        s != 0 && (db.id2slot[id] = remap[s])
    end
    db.dups = Dict(remap[s] => v for (s, v) in db.dups)
    for (h, s) in db.hash2slot
        db.hash2slot[h] = remap[s]
    end
    for (h, v) in db.collisions
        map!(s -> remap[s], v, v)
    end
    db
end
