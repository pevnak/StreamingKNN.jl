# ---------------------------------------------------------------------------
# workspace

mutable struct Workspace{C}
    xbuf::Matrix{C}        # dim × chunksize   (converted chunk; empty when T == C)
    hbuf::Matrix{C}        # ndims × chunksize (converted head chunk; empty without filter or when T == C)
    sbuf::Matrix{C}        # max(chunksize, maxB) × maxB  (scores of one chunk or of the self block)
    stbuf::Matrix{C}       # maxB × max(chunksize, maxB)  (transposed head scores; empty without filter)
    thr::Vector{C}         # per query: heap threshold frozen for one chunk
    qbuf::Matrix{C}        # dim × maxB        (converted queries)
    gbuf::Matrix{C}        # ndims × maxB      (query heads)
    qn2::Vector{C}         # squared query norms
    qnorm::Vector{C}       # query norms
    qtail_c::Vector{C}     # query norms beyond the coarse sub-head
    qtail_g::Vector{C}     # query norms beyond the gemm sub-head
    qtail_h::Vector{C}     # query norms beyond the stored head
    ubheaps::Vector{TopK{C}}   # threshold pass: best coarse bounds per query
    nref::Vector{Int}      # per query: exact scores computed in the current chunk
    heaps::Vector{TopK{C}}
end

"Return a workspace able to serve a batch of `B` queries with `k` neighbours."
function workspace!(db::KNNDatabase{T,C}, B::Int, k::Int) where {T,C}
    st = db.storage
    ws = db.ws
    if ws === nothing || size(ws.qbuf, 2) < B || length(ws.heaps) < B || ws.heaps[1].k != k
        maxB = ws === nothing ? B : max(B, size(ws.qbuf, 2))
        nd = head_dims(db.filter)
        xbuf = T === C ? Matrix{C}(undef, 0, 0) :
               (ws === nothing ? Matrix{C}(undef, st.dim, st.chunksize) : ws.xbuf)
        hbuf = (T === C || nd == 0) ? Matrix{C}(undef, 0, 0) :
               (ws === nothing ? Matrix{C}(undef, nd, st.chunksize) : ws.hbuf)
        ws = Workspace{C}(xbuf, hbuf,
                          Matrix{C}(undef, max(st.chunksize, maxB), maxB),
                          nd == 0 ? Matrix{C}(undef, 0, 0) : Matrix{C}(undef, maxB, max(st.chunksize, maxB)),
                          Vector{C}(undef, maxB),
                          Matrix{C}(undef, st.dim, maxB),
                          Matrix{C}(undef, nd, maxB),
                          Vector{C}(undef, maxB), Vector{C}(undef, maxB), Vector{C}(undef, maxB),
                          Vector{C}(undef, maxB), Vector{C}(undef, maxB),
                          [TopK{C}(max(1, ncold(db.filter))) for _ in 1:maxB],
                          zeros(Int, maxB),
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

"View of chunk `c` of `storage` (first `ncols` columns) with element type `C`, converting into `buf` if needed."
@inline function chunk_as(::Type{C}, storage::ChunkedStorage{T}, buf::Matrix{C}, c::Int, ncols::Int) where {T,C}
    chunk = storage.chunks[c]
    T === C && return view(chunk, :, 1:ncols)
    convert_columns!(buf, chunk, ncols)
    view(buf, :, 1:ncols)
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

"Transposed direct kernel: `St[b, j] = ⟨x_j, q_b⟩` with on-the-fly conversion."
function direct_scores_t!(St::AbstractMatrix{C}, chunk::AbstractMatrix, ncols::Int, Q::AbstractMatrix{C},
                          d::Int = size(Q, 1)) where {C}
    B = size(Q, 2)
    Threads.@threads :static for j in 1:ncols
        @inbounds for b in 1:B
            s = zero(C)
            @simd for i in 1:d
                s += C(chunk[i, j]) * Q[i, b]
            end
            St[b, j] = s
        end
    end
    St
end

"Scores of the first `ncols` columns of chunk `c` of `storage` against `Q`, into `S`."
function chunk_scores!(S::AbstractMatrix{C}, storage::ChunkedStorage{T}, buf::Matrix{C}, c::Int, ncols::Int,
                       Q::AbstractMatrix{C}, direct::Bool) where {T,C}
    if direct
        direct_scores!(S, storage.chunks[c], ncols, Q)
    else
        X = chunk_as(C, storage, buf, c, ncols)
        mul!(S, transpose(X), Q)
    end
    S
end

"Exact inner product between stored slot `s` and the query vector `q`."
@inline function exact_score(storage::ChunkedStorage, s::Int, q::AbstractVector{C}) where {C}
    x = storage[s]
    acc = zero(C)
    @inbounds @simd for i in eachindex(x)
        acc += C(x[i]) * q[i]
    end
    acc
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

"Exact scores to compute before trusting the bound when a heap is still empty."
cold_start_size(k::Int) = max(2k, 64)

ncold(::NoFilter) = 0
ncold(f::HeadFilter) = f.ncold

"""
Queries handled together by one thread in the filtered scan.  The block's
query columns (`dim × qblock` Float32) should stay in cache while the chunk
is scanned; 256 queries of 4096 dimensions are 4 MB.  Smaller batches use
smaller blocks so that every thread gets one.
"""
const QBLOCK = 256
function query_block(B::Int)
    qb = clamp(B ÷ Threads.nthreads(), 16, QBLOCK)
    qb, cld(B, qb)
end

"Convert stored slot `sl` into the Float32 buffer `xf`."
@inline function load_vector!(xf::Vector{C}, storage::ChunkedStorage, sl::Int) where {C}
    x = storage[sl]
    @inbounds @simd for i in eachindex(xf)
        xf[i] = C(x[i])
    end
    xf
end

@inline function dot_columns(xf::Vector{C}, Q::AbstractMatrix{C}, b::Int) where {C}
    acc = zero(C)
    @inbounds @simd for i in eachindex(xf)
        acc += xf[i] * Q[i, b]
    end
    acc
end

@inline inheap(h::TopK, sl::Int) = any(==(sl), view(h.ids, 1:h.n))

"Convert head coordinates `gemm+1:ndims` of slot `sl` into `hf`."
@inline function load_head_tail!(hf::Vector{C}, head::ChunkedStorage, sl::Int, gemm::Int) where {C}
    h = head[sl]
    @inbounds @simd for i in eachindex(hf)
        hf[i] = C(h[gemm + i])
    end
    hf
end

"Partial inner product of the stage-2 head segment with query head `b`."
@inline function dot_head_tail(hf::Vector{C}, G::AbstractMatrix{C}, b::Int, gemm::Int) where {C}
    acc = zero(C)
    @inbounds @simd for i in eachindex(hf)
        acc += hf[i] * G[gemm + i, b]
    end
    acc
end

"""
Filtered scan of one block of slots against one block of queries.

`St[b, j]` holds the stage-1 head scores of slot `off + j`.  For every
(slot, query) pair the stage-1 bound is compared with the query's heap
threshold; survivors are bounded again with the remaining stored head
coordinates (when the filter has them) and only then scored exactly from
the full vector.  Thresholds tighten as the scan proceeds.  Queries whose
heap is not full yet first refine their `cold_start_size(k)` best bounds.
`firstb[j]`, when given, is the first query allowed to see slot `off + j`
(causal mask of the self block).

Returns `(stage2, refined)` counts.
"""
function scan_block!(db::KNNDatabase{T,C}, ws::Workspace{C}, St::AbstractMatrix{C}, Q::AbstractMatrix{C},
                     off::Int, ncols::Int, b0::Int, b1::Int, firstb::Union{Nothing,Vector{Int}}) where {T,C}
    f = db.filter
    heaps, scale, shift = ws.heaps, db.scale, db.shift
    tail_g, tail_h, headerr, storage, head = db.tail_g, db.tail_h, db.headerr, db.storage, db.head
    qtail_g, qtail_h, qnorm = ws.qtail_g, ws.qtail_h, ws.qnorm
    G = view(ws.gbuf, :, 1:size(Q, 2))
    gemm = f.gemm
    stage2 = has_stage2(f)
    xf = Vector{C}(undef, storage.dim)
    hf = Vector{C}(undef, stage2 ? f.ndims - gemm : 0)
    nb = b1 - b0 + 1
    n2count = 0
    count = 0
    # --- cold start: queries with a non-full heap refine their best bounds first
    @inbounds if any(b -> heaps[b].n < heaps[b].k, b0:b1)
        ubs = [heaps[b].n < heaps[b].k ? TopK{C}(cold_start_size(heaps[b].k)) : nothing for b in b0:b1]
        for j in 1:ncols
            sl = off + j
            sc, sh, tl, he = scale[sl], shift[sl], tail_g[sl], headerr[sl]
            bstart = firstb === nothing ? b0 : max(b0, firstb[j])
            for b in bstart:b1
                ub = ubs[b - b0 + 1]
                ub === nothing && continue
                key = muladd(sc, St[b, j] + qtail_g[b] * tl + qnorm[b] * he, sh)
                key > threshold(ub) && push!(ub, key, sl)
            end
        end
        for i in 1:nb
            ub = ubs[i]
            ub === nothing && continue
            b = b0 + i - 1
            h = heaps[b]
            for r in 1:ub.n
                sl = ub.ids[r]
                inheap(h, sl) && continue
                key = muladd(scale[sl], dot_columns(load_vector!(xf, storage, sl), Q, b), shift[sl])
                count += 1
                key > threshold(h) && push!(h, key, sl)
            end
        end
    end
    # --- bounded scan with inline refinement
    thr = Vector{C}(undef, nb)
    ubk = Vector{C}(undef, nb)
    @inbounds for i in 1:nb
        thr[i] = threshold(heaps[b0 + i - 1])
    end
    @inbounds for j in 1:ncols
        sl = off + j
        sc, sh, tl, he = scale[sl], shift[sl], tail_g[sl], headerr[sl]
        sh == typemin(C) && continue                          # dead slot
        istart = firstb === nothing ? 1 : max(1, firstb[j] - b0 + 1)
        istart <= nb || continue
        hits = 0
        @simd for i in istart:nb
            b = b0 + i - 1
            v = muladd(sc, St[b, j] + qtail_g[b] * tl + qnorm[b] * he, sh)
            ubk[i] = v
            hits += ifelse(v > thr[i], 1, 0)
        end
        hits == 0 && continue
        loaded = false
        hloaded = false
        for i in istart:nb
            b = b0 + i - 1
            ubk[i] > thr[i] || continue
            if stage2
                if !hloaded
                    load_head_tail!(hf, head, sl, gemm)
                    hloaded = true
                end
                n2count += 1
                ub2 = muladd(sc, St[b, j] + dot_head_tail(hf, G, b, gemm) + qtail_h[b] * tail_h[sl] + qnorm[b] * he, sh)
                ub2 > thr[i] || continue
            end
            if !loaded
                load_vector!(xf, storage, sl)
                loaded = true
            end
            key = muladd(sc, dot_columns(xf, Q, b), sh)
            count += 1
            if key > thr[i]
                h = heaps[b]
                inheap(h, sl) && continue
                push!(h, key, sl)
                thr[i] = threshold(h)
            end
        end
    end
    n2count, count
end

"""
    scan_bounds!(db, ws, St, Q, off, ncols, B, firstb = nothing)

Distribute the filtered scan of one chunk over threads: blocks of queries
when there are enough of them, otherwise blocks of slots whose candidates
are merged at the end.
"""
function scan_bounds!(db::KNNDatabase{T,C}, ws::Workspace{C}, St::AbstractMatrix{C}, Q::AbstractMatrix{C},
                      off::Int, ncols::Int, B::Int, firstb::Union{Nothing,Vector{Int}} = nothing) where {T,C}
    qb, nblocks = query_block(B)
    st = db.stats
    if nblocks >= Threads.nthreads()
        counts = Vector{Tuple{Int,Int}}(undef, nblocks)
        Threads.@threads :static for blk in 1:nblocks
            b0 = (blk - 1) * qb + 1
            b1 = min(B, blk * qb)
            counts[blk] = scan_block!(db, ws, St, Q, off, ncols, b0, b1, firstb)
        end
        st.stage2 += sum(first, counts)
        st.refined += sum(last, counts)
    else
        n2, nr = scan_slot_major!(db, ws, St, Q, off, ncols, B, firstb)
        st.stage2 += n2
        st.refined += nr
    end
    st.scanned += ncols * B
    ws.heaps
end

"Slot-major variant for small batches: threads split the slots, candidates are merged at the end."
function scan_slot_major!(db::KNNDatabase{T,C}, ws::Workspace{C}, St::AbstractMatrix{C}, Q::AbstractMatrix{C},
                          off::Int, ncols::Int, B::Int, firstb::Union{Nothing,Vector{Int}}) where {T,C}
    f = db.filter
    heaps, scale, shift = ws.heaps, db.scale, db.shift
    tail_g, tail_h, headerr, storage, head = db.tail_g, db.tail_h, db.headerr, db.storage, db.head
    qtail_g, qtail_h, qnorm, thr = ws.qtail_g, ws.qtail_h, ws.qnorm, ws.thr
    G = view(ws.gbuf, :, 1:B)
    gemm = f.gemm
    stage2 = has_stage2(f)
    refined = 0
    # cold start (B is small, the strided access to St is cheap here)
    @inbounds for b in 1:B
        h = heaps[b]
        if h.n < h.k
            xf = Vector{C}(undef, storage.dim)
            ub = TopK{C}(cold_start_size(h.k))
            for j in 1:ncols
                firstb !== nothing && firstb[j] > b && continue
                sl = off + j
                key = muladd(scale[sl], St[b, j] + qtail_g[b] * tail_g[sl] + qnorm[b] * headerr[sl], shift[sl])
                key > threshold(ub) && push!(ub, key, sl)
            end
            for r in 1:ub.n
                sl = ub.ids[r]
                inheap(h, sl) && continue
                key = muladd(scale[sl], dot_columns(load_vector!(xf, storage, sl), Q, b), shift[sl])
                refined += 1
                key > threshold(h) && push!(h, key, sl)
            end
        end
        thr[b] = threshold(h)
    end
    nt = Threads.nthreads()
    found = [Tuple{Int,Int,C}[] for _ in 1:nt]
    counts = Vector{Tuple{Int,Int}}(undef, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * ncols ÷ nt + 1
        hi = t * ncols ÷ nt
        xf = Vector{C}(undef, storage.dim)
        hf = Vector{C}(undef, stage2 ? f.ndims - gemm : 0)
        out = found[t]
        cnt = 0
        n2count = 0
        @inbounds for j in lo:hi
            sl = off + j
            sc, sh, tl, he = scale[sl], shift[sl], tail_g[sl], headerr[sl]
            sh == typemin(C) && continue
            loaded = false
            hloaded = false
            for b in (firstb === nothing ? 1 : firstb[j]):B
                ubkey = muladd(sc, St[b, j] + qtail_g[b] * tl + qnorm[b] * he, sh)
                ubkey > thr[b] || continue
                if stage2
                    if !hloaded
                        load_head_tail!(hf, head, sl, gemm)
                        hloaded = true
                    end
                    n2count += 1
                    ub2 = muladd(sc, St[b, j] + dot_head_tail(hf, G, b, gemm) + qtail_h[b] * tail_h[sl] + qnorm[b] * he, sh)
                    ub2 > thr[b] || continue
                end
                if !loaded
                    load_vector!(xf, storage, sl)
                    loaded = true
                end
                cnt += 1
                key = muladd(sc, dot_columns(xf, Q, b), sh)
                key > thr[b] && push!(out, (b, sl, key))
            end
        end
        counts[t] = (n2count, cnt)
    end
    @inbounds for out in found, (b, sl, key) in out
        h = heaps[b]
        key > threshold(h) || continue
        inheap(h, sl) && continue
        push!(h, key, sl)
    end
    sum(first, counts), refined + sum(last, counts)
end

"""
Threshold pass: bound every slot in `1:hi` with the coarse sub-head, keep the
`ncold` best per query, score those exactly and seed the heaps, so that the
main pass starts with thresholds close to the final k-th scores.
"""
function threshold_pass!(db::KNNDatabase{T,C}, ws::Workspace{C}, Q::AbstractMatrix{C}, hi::Int, direct::Bool) where {T,C}
    f = db.filter
    nc = f.coarse
    (nc == 0 || hi == 0) && return
    st = db.storage
    B = size(Q, 2)
    Gc = view(ws.gbuf, 1:nc, 1:B)
    scale, shift, tailc, headerr = db.scale, db.shift, db.tail_c, db.headerr
    qtc, qnorm, ubheaps, heaps = ws.qtail_c, ws.qnorm, ws.ubheaps, ws.heaps
    for b in 1:B
        reset!(ubheaps[b])
    end
    qb, nblocks = query_block(B)
    for c in 1:nchunks(st)
        off = chunk_offset(st, c)
        ncols = min(st.chunksize, hi - off)
        ncols <= 0 && break
        St = view(ws.stbuf, 1:B, 1:ncols)
        if direct
            direct_scores_t!(St, db.head.chunks[c], ncols, Gc, nc)
        else
            Hc = chunk_as(C, db.head, ws.hbuf, c, ncols)
            mul!(St, transpose(Gc), view(Hc, 1:nc, :))
        end
        Threads.@threads :static for blk in 1:nblocks
            b0 = (blk - 1) * qb + 1
            b1 = min(B, blk * qb)
            @inbounds for j in 1:ncols
                sl = off + j
                sc, sh, tl, he = scale[sl], shift[sl], tailc[sl], headerr[sl]
                sh == typemin(C) && continue
                for b in b0:b1
                    key = muladd(sc, St[b, j] + qtc[b] * tl + qnorm[b] * he, sh)
                    ub = ubheaps[b]
                    key > threshold(ub) && push!(ub, key, sl)
                end
            end
        end
    end
    nref = ws.nref
    Threads.@threads :static for b in 1:B
        xf = Vector{C}(undef, st.dim)
        ub = ubheaps[b]
        h = heaps[b]
        @inbounds for r in 1:ub.n
            sl = ub.ids[r]
            key = muladd(scale[sl], dot_columns(load_vector!(xf, st, sl), Q, b), shift[sl])
            key > threshold(h) && !inheap(h, sl) && push!(h, key, sl)
        end
        nref[b] = ub.n
    end
    db.stats.refined += sum(view(nref, 1:B))
    nothing
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
    filtered = hasfilter(db)
    G = view(ws.gbuf, :, 1:B)
    filtered && threshold_pass!(db, ws, Q, hi, direct)
    for c in 1:nchunks(st)
        off = chunk_offset(st, c)
        ncols = min(st.chunksize, hi - off)
        ncols <= 0 && break
        if filtered
            ng = db.filter.gemm
            St = view(ws.stbuf, 1:B, 1:ncols)
            if direct
                direct_scores_t!(St, db.head.chunks[c], ncols, view(G, 1:ng, :), ng)
            else
                Hc = chunk_as(C, db.head, ws.hbuf, c, ncols)
                mul!(St, transpose(view(G, 1:ng, :)), view(Hc, 1:ng, :))
            end
            scan_bounds!(db, ws, St, Q, off, ncols, B)
        else
            S = view(ws.sbuf, 1:ncols, 1:B)
            chunk_scores!(S, st, ws.xbuf, c, ncols, Q, direct)
            scan_scores!(ws.heaps, S, db.scale, db.shift, off, ncols, B)
        end
    end
    ws
end

"""
    search_new_slots!(db, ws, Q, n0, newpos)

Causal self block of a streaming batch.  Slot `n0 + j` was created by the
batch member at position `newpos[j]` and is visible only to later members.
Uses the filter when there is one, with the causal mask applied to the
bounds; otherwise one exact matrix product (the block is at most B × B).
"""
function search_new_slots!(db::KNNDatabase{T,C}, ws::Workspace{C}, Q::AbstractMatrix{C},
                           n0::Int, newpos::Vector{Int}) where {T,C}
    nnew = length(newpos)
    nnew == 0 && return ws
    B = size(Q, 2)
    if hasfilter(db)
        f = db.filter
        Hn = Matrix{C}(undef, f.gemm, nnew)
        for j in 1:nnew
            @inbounds copyto!(view(Hn, :, j), view(db.head[n0 + j], 1:f.gemm))
        end
        St = view(ws.stbuf, 1:B, 1:nnew)
        mul!(St, transpose(view(ws.gbuf, 1:f.gemm, 1:B)), Hn)
        firstb = newpos .+ 1
        scan_bounds!(db, ws, St, Q, n0, nnew, B, firstb)
        return ws
    end
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

"Load queries into the workspace: convert to `C`, compute norms and heads, reset heaps."
function load_queries!(db::KNNDatabase{T,C}, ws::Workspace{C}, X::AbstractMatrix) where {T,C}
    d, B = size(X)
    Q = view(ws.qbuf, :, 1:B)
    convert_columns!(Q, X, B)
    @inbounds for b in 1:B
        ws.qn2[b] = squared_norm(C, view(Q, :, b))
        ws.qnorm[b] = sqrt(ws.qn2[b])
    end
    if hasfilter(db)
        G = view(ws.gbuf, :, 1:B)
        heads!(G, db.filter, Q)
        nc, ng = db.filter.coarse, db.filter.gemm
        @inbounds for b in 1:B
            ws.qtail_c[b] = sqrt(max(zero(C), ws.qn2[b] - squared_norm(C, view(G, 1:nc, b))))
            ws.qtail_g[b] = sqrt(max(zero(C), ws.qn2[b] - squared_norm(C, view(G, 1:ng, b))))
            ws.qtail_h[b] = sqrt(max(zero(C), ws.qn2[b] - squared_norm(C, view(G, :, b))))
        end
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
        Q = load_queries!(db, ws, Xb)
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
    Q = load_queries!(db, ws, Xb)
    n0 = nslots(db)
    # insert first (with dedup); causality is restored by the masked self block
    newpos = insert_batch!(db, Xb)
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
