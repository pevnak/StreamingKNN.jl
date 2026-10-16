"""
    TopK{C}(k)

Bounded min-heap keeping the `k` entries with the largest keys seen so far.
`keys[1]` is always the smallest retained key, i.e. the admission threshold
once the heap is full.
"""
mutable struct TopK{C<:AbstractFloat}
    k::Int
    n::Int
    keys::Vector{C}
    ids::Vector{Int}
    perm::Vector{Int}
end

TopK{C}(k::Integer) where {C} =
    TopK{C}(k, 0, Vector{C}(undef, k), Vector{Int}(undef, k), Vector{Int}(undef, k))

@inline reset!(h::TopK) = (h.n = 0; h)

"Current admission threshold: a key must be strictly greater to enter."
@inline threshold(h::TopK{C}) where {C} = h.n < h.k ? typemin(C) : @inbounds h.keys[1]

@inline function Base.push!(h::TopK, key, id::Int)
    @inbounds if h.n < h.k
        key > typemin(key) || return h
        h.n += 1
        h.keys[h.n] = key
        h.ids[h.n] = id
        _siftup!(h, h.n)
    elseif key > h.keys[1]
        h.keys[1] = key
        h.ids[1] = id
        _siftdown!(h, 1)
    end
    h
end

@inline function _swap!(h::TopK, i, j)
    @inbounds begin
        h.keys[i], h.keys[j] = h.keys[j], h.keys[i]
        h.ids[i], h.ids[j] = h.ids[j], h.ids[i]
    end
end

function _siftup!(h::TopK, i::Int)
    @inbounds while i > 1
        p = i >> 1
        h.keys[p] <= h.keys[i] && break
        _swap!(h, i, p)
        i = p
    end
end

function _siftdown!(h::TopK, i::Int)
    n = h.n
    @inbounds while true
        l = 2i
        l > n && break
        r = l + 1
        c = (r <= n && h.keys[r] < h.keys[l]) ? r : l
        h.keys[c] >= h.keys[i] && break
        _swap!(h, i, c)
        i = c
    end
end

"""
    sorted_indices!(h) -> view

Indices into `h.keys`/`h.ids` ordered by decreasing key.
"""
function sorted_indices!(h::TopK)
    n = h.n
    p = view(h.perm, 1:n)
    p .= 1:n
    sort!(p; by = i -> @inbounds(h.keys[i]), rev = true)
    p
end
