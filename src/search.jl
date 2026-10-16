# ---------------------------------------------------------------------------
# workspace

mutable struct Workspace{C}
    xbuf::Matrix{C}        # dim × chunksize   (converted chunk; empty when T == C)
    sbuf::Matrix{C}        # max(chunksize, maxB) × maxB  (scores of one chunk or of the self block)
    qbuf::Matrix{C}        # dim × maxB        (converted queries)
    qn2::Vector{C}         # squared query norms
    heaps::Vector{TopK{C}}
end

"Return a workspace able to serve a batch of `B` queries with `k` neighbours."
function workspace!(db::KNNDatabase{T,C}, B::Int, k::Int) where {T,C}
    st = db.storage
    ws = db.ws
    if ws === nothing || size(ws.qbuf, 2) < B || length(ws.heaps) < B || ws.heaps[1].k != k
        maxB = ws === nothing ? B : max(B, size(ws.qbuf, 2))
        xbuf = T === C ? Matrix{C}(undef, 0, 0) :
               (ws === nothing ? Matrix{C}(undef, st.dim, st.chunksize) : ws.xbuf)
        ws = Workspace{C}(xbuf,
                          Matrix{C}(undef, max(st.chunksize, maxB), maxB),
                          Matrix{C}(undef, st.dim, maxB),
                          Vector{C}(undef, maxB),
                          [TopK{C}(k) for _ in 1:maxB])
        db.ws = ws
    end
    ws::Workspace{C}
end

# ---------------------------------------------------------------------------
# kernels

"Threaded conversion of the first `ncols` columns of `src` into `dst`."
function convert_columns!(dst::AbstractMatrix{C}, src::AbstractMatrix, ncols::Int) where {C}
    d = size(src, 1)
    Threads.@threads :static for j in 1:ncols
        @inbounds @simd for i in 1:d
            dst[i, j] = C(src[i, j])
        end
    end
    dst
end

"View of chunk `c` (first `ncols` columns) with element type `C`."
@inline function chunk_as(::Type{C}, db::KNNDatabase{T,C}, ws::Workspace{C}, c::Int, ncols::Int) where {T,C}
    chunk = db.storage.chunks[c]
    T === C && return view(chunk, :, 1:ncols)
    convert_columns!(ws.xbuf, chunk, ncols)
    view(ws.xbuf, :, 1:ncols)
end

"""
Direct kernel for small batches: `S[j, b] = ⟨x_j, q_b⟩` computed with
on-the-fly conversion, avoiding the Float32 copy of the whole chunk.
"""
function direct_scores!(S::AbstractMatrix{C}, chunk::AbstractMatrix, ncols::Int, Q::AbstractMatrix{C}) where {C}
    d, B = size(Q)
    Threads.@threads :static for j in 1:ncols
        @inbounds for b in 1:B
            s = zero(C)
            @simd for i in 1:d
                s += C(chunk[i, j]) * Q[i, b]
            end
            S[j, b] = s
        end
    end
    S
end

"Feed the scores of slots `off+1 .. off+ncols` into the heaps."
function scan_scores!(heaps::Vector{TopK{C}}, S::AbstractMatrix{C},
                      scale::Vector{C}, shift::Vector{C}, off::Int, ncols::Int, B::Int) where {C}
    Threads.@threads :static for b in 1:B
        h = heaps[b]
        thr = threshold(h)
        @inbounds for j in 1:ncols
            s = off + j
            key = muladd(scale[s], S[j, b], shift[s])
            if key > thr
                push!(h, key, s)
                thr = threshold(h)
            end
        end
    end
    heaps
end

"""
    search_slots!(db, ws, Q, hi)

Score the queries `Q` (dim × B, element type `C`) against slots `1:hi` and
update the heaps of `ws`.
"""
function search_slots!(db::KNNDatabase{T,C}, ws::Workspace{C}, Q::AbstractMatrix{C}, hi::Int) where {T,C}
    st = db.storage
    B = size(Q, 2)
    direct = T !== C && B <= db.direct_batch
    for c in 1:nchunks(st)
        off = chunk_offset(st, c)
        ncols = min(st.chunksize, hi - off)
        ncols <= 0 && break
        S = view(ws.sbuf, 1:ncols, 1:B)
        if direct
            direct_scores!(S, st.chunks[c], ncols, Q)
        else
            X = chunk_as(C, db, ws, c, ncols)
            mul!(S, transpose(X), Q)
        end
        scan_scores!(ws.heaps, S, db.scale, db.shift, off, ncols, B)
    end
    ws
end

"""
    search_new_slots!(db, ws, Q, n0, newpos)

Causal self block of a streaming batch.  Slot `n0 + j` was created by the
batch member at position `newpos[j]` and is visible only to later members.
"""
function search_new_slots!(db::KNNDatabase{T,C}, ws::Workspace{C}, Q::AbstractMatrix{C},
                           n0::Int, newpos::Vector{Int}) where {T,C}
    nnew = length(newpos)
    nnew == 0 && return ws
    B = size(Q, 2)
    st = db.storage
    Xn = Matrix{C}(undef, st.dim, nnew)
    for j in 1:nnew
        @inbounds copyto!(view(Xn, :, j), st[n0 + j])
    end
    Sn = view(ws.sbuf, 1:nnew, 1:B)
    mul!(Sn, transpose(Xn), Q)
    scale, shift, heaps = db.scale, db.shift, ws.heaps
    Threads.@threads :static for b in 1:B
        h = heaps[b]
        thr = threshold(h)
        @inbounds for j in 1:nnew
            newpos[j] < b || break
            s = n0 + j
            key = muladd(scale[s], Sn[j, b], shift[s])
            if key > thr
                push!(h, key, s)
                thr = threshold(h)
            end
        end
    end
    ws
end

"Load queries into the workspace: convert to `C` and compute squared norms."
function load_queries!(ws::Workspace{C}, X::AbstractMatrix) where {C}
    d, B = size(X)
    Q = view(ws.qbuf, :, 1:B)
    convert_columns!(Q, X, B)
    @inbounds for b in 1:B
        ws.qn2[b] = squared_norm(C, view(Q, :, b))
    end
    for b in 1:B
        reset!(ws.heaps[b])
    end
    Q
end

"""
    refine_scores!(metric, scores, db, slots, q, n)

Recompute the scores of the `n` selected slots directly from storage.  Only
`Euclidean` needs this: the expansion `‖q‖² + ‖x‖² - 2⟨x,q⟩` evaluated in
`C` loses all digits when the distance is tiny compared to the norms (e.g.
near-duplicates).  Costs `n * dim` per query, negligible next to the scan.
"""
@inline refine_scores!(::Metric, scores, db, slots, q, n) = nothing
function refine_scores!(::Euclidean, scores::AbstractVector{C}, db::KNNDatabase, slots, q, n) where {C}
    st = db.storage
    @inbounds for r in 1:n
        x = st[slots[r]]
        s = zero(C)
        @simd for i in eachindex(x)
            d = C(x[i]) - q[i]
            s += d * d
        end
        scores[r] = sqrt(s)
    end
    # exact values may reorder near-ties
    if n > 1 && !issorted(view(scores, 1:n))
        p = sortperm(view(scores, 1:n))
        scores[1:n] = scores[p]
        slots[1:n] = slots[p]
    end
    nothing
end

"Write sorted results of the heaps into `ids`/`scores` (k × B), padding with 0 / NaN."
function collect_results!(ids::AbstractMatrix{Int}, scores::AbstractMatrix{C},
                          db::KNNDatabase{T,C}, ws::Workspace{C}, Q::AbstractMatrix{C}, B::Int) where {T,C}
    k = size(ids, 1)
    m = db.metric
    @inbounds for b in 1:B
        h = ws.heaps[b]
        perm = sorted_indices!(h)
        qn2 = ws.qn2[b]
        n = h.n
        slots = view(ids, 1:n, b)          # temporarily hold slots, replaced by ids below
        for r in 1:n
            p = perm[r]
            slots[r] = h.ids[p]
            scores[r, b] = finalscore(m, h.keys[p], qn2)
        end
        refine_scores!(m, view(scores, :, b), db, slots, view(Q, :, b), n)
        for r in 1:n
            ids[r, b] = db.rep_id[slots[r]]
        end
        for r in n+1:k
            ids[r, b] = 0
            scores[r, b] = C(NaN)
        end
    end
    ids, scores
end

# ---------------------------------------------------------------------------
# public API

function _check_dim(db, X)
    size(X, 1) == dimension(db) ||
        throw(DimensionMismatch("database holds $(dimension(db))-dimensional vectors, got $(size(X, 1))"))
end

"""
    knn(db, X; k = db.k, batchsize = 1024) -> (ids, scores)

Exact `k` nearest neighbours of every column of `X` among the alive vectors
of `db`.  Both outputs are `k × size(X, 2)` matrices, best neighbour first.
Rows beyond the number of available vectors hold id `0` and score `NaN`.
Scores are inner products, cosines, or Euclidean distances according to the
metric of `db`.

    knn(db, x::AbstractVector; k) -> (ids::Vector, scores::Vector)
"""
function knn(db::KNNDatabase{T,C}, X::AbstractMatrix; k::Integer = db.k, batchsize::Integer = 1024) where {T,C}
    _check_dim(db, X)
    n = size(X, 2)
    ids = Matrix{Int}(undef, k, n)
    scores = Matrix{C}(undef, k, n)
    for lo in 1:batchsize:n
        hi = min(lo + batchsize - 1, n)
        Xb = view(X, :, lo:hi)
        B = hi - lo + 1
        ws = workspace!(db, B, Int(k))
        Q = load_queries!(ws, Xb)
        search_slots!(db, ws, Q, nslots(db))
        collect_results!(view(ids, :, lo:hi), view(scores, :, lo:hi), db, ws, Q, B)
    end
    ids, scores
end

function knn(db::KNNDatabase, x::AbstractVector; k::Integer = db.k)
    ids, scores = knn(db, reshape(x, :, 1); k)
    vec(ids), vec(scores)
end

"""
    stream_knn!(db, X; k = db.k, batchsize = 1024) -> (ids, scores)

Process the columns of `X` as a time-ordered stream.  For every column `t`
the result holds the `k` nearest neighbours among everything inserted before
it (the initial content of `db` and columns `1:t-1`), and then column `t` is
inserted.  Equivalent to calling [`knn_push!`](@ref) column by column, but
batched so the work is done with large matrix products.

Ids reported are the representative ids of the matched vectors (see
[`duplicates`](@ref)).  The ids assigned to the columns of `X` are
`nseen(db) + 1 : nseen(db) + size(X, 2)` as evaluated before the call.
"""
function stream_knn!(db::KNNDatabase{T,C}, X::AbstractMatrix; k::Integer = db.k, batchsize::Integer = 1024) where {T,C}
    _check_dim(db, X)
    n = size(X, 2)
    ids = Matrix{Int}(undef, k, n)
    scores = Matrix{C}(undef, k, n)
    for lo in 1:batchsize:n
        hi = min(lo + batchsize - 1, n)
        stream_batch!(view(ids, :, lo:hi), view(scores, :, lo:hi), db, view(X, :, lo:hi), Int(k))
    end
    ids, scores
end

function stream_batch!(ids::AbstractMatrix{Int}, scores::AbstractMatrix{C},
                       db::KNNDatabase{T,C}, Xb::AbstractMatrix, k::Int) where {T,C}
    B = size(Xb, 2)
    ws = workspace!(db, B, k)
    Q = load_queries!(ws, Xb)
    n0 = nslots(db)
    # insert first (with dedup); causality is restored by the masked self block
    newpos = Int[]
    xt = Vector{T}(undef, dimension(db))
    for i in 1:B
        xt .= view(Xb, :, i)
        _, isnew = insert_vector!(db, xt)
        isnew && push!(newpos, i)
    end
    search_slots!(db, ws, Q, n0)
    search_new_slots!(db, ws, Q, n0, newpos)
    collect_results!(ids, scores, db, ws, Q, B)
end

"""
    knn_push!(db, x; k = db.k) -> (ids, scores)

One streaming step: neighbours of `x` among the current content, then insert `x`.
"""
function knn_push!(db::KNNDatabase, x::AbstractVector; k::Integer = db.k)
    ids, scores = stream_knn!(db, reshape(x, :, 1); k)
    vec(ids), vec(scores)
end
