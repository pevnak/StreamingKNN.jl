"""
    KNNDatabase{T,C,M}

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
mutable struct KNNDatabase{T,C<:AbstractFloat,M<:Metric}
    metric::M
    k::Int
    storage::ChunkedStorage{T}
    direct_batch::Int                 # use the direct dot kernel for batches up to this size
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
    KNNDatabase{T}(dim; metric = InnerProduct(), k = 10, chunksize = 2^15, compute = ..., direct_batch = 16)
    KNNDatabase(X::AbstractMatrix; kwargs...)

Create an empty database for `dim`-dimensional vectors stored as `T`, or one
initialised with the columns of `X` (stored as `T = eltype(X)` unless given).
"""
function KNNDatabase{T}(dim::Integer;
                        metric::Metric = InnerProduct(),
                        k::Integer = 10,
                        chunksize::Integer = 1 << 15,
                        compute::Type{C} = default_compute_type(T),
                        direct_batch::Integer = 16) where {T,C<:AbstractFloat}
    k >= 1 || throw(ArgumentError("k must be positive"))
    KNNDatabase{T,C,typeof(metric)}(
        metric, Int(k), ChunkedStorage{T}(dim; chunksize), Int(direct_batch),
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
          "unique=$(length(db)), ids=$(nseen(db)), slots=$(nslots(db)))")
end

# ---------------------------------------------------------------------------
# hashing & dedup

"Hash of a vector consistent with elementwise `isequal`."
function vhash(x::AbstractVector)
    h = hash(length(x))
    @inbounds for i in eachindex(x)
        h = hash(x[i], h)
    end
    h
end

function vequal(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        isequal(a[i], b[i]) || return false
    end
    true
end

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

Assign a new id to `xt` (already converted to the storage type).  Returns the
slot the id was attached to and whether that slot was created by this call.
"""
function insert_vector!(db::KNNDatabase{T,C}, xt::AbstractVector{T}) where {T,C}
    h = vhash(xt)
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
    n2 = squared_norm(C, xt)
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
    push!(db, x) -> id

Insert one vector without searching.  Returns the id assigned to it.
"""
function Base.push!(db::KNNDatabase{T}, x::AbstractVector) where {T}
    length(x) == dimension(db) || throw(DimensionMismatch("expected length $(dimension(db)), got $(length(x))"))
    xt = Vector{T}(undef, length(x))
    xt .= x
    insert_vector!(db, xt)
    db.nseen
end

"""
    append!(db, X) -> ids

Insert the columns of `X` without searching.  Returns the assigned ids.
"""
function Base.append!(db::KNNDatabase{T}, X::AbstractMatrix) where {T}
    size(X, 1) == dimension(db) || throw(DimensionMismatch("expected $(dimension(db)) rows, got $(size(X, 1))"))
    first = db.nseen + 1
    xt = Vector{T}(undef, size(X, 1))
    for j in axes(X, 2)
        xt .= view(X, :, j)
        insert_vector!(db, xt)
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
        db.scale[w] = db.scale[r]
        db.shift[w] = db.shift[r]
        db.norms2[w] = db.norms2[r]
        db.rep_id[w] = db.rep_id[r]
    end
    truncate!(db.storage, w)
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
