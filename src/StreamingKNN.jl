"""
    StreamingKNN

Exact k-nearest-neighbour search for streaming vector data: query-then-insert
on a growing database of up to millions of high-dimensional vectors, with
duplicate folding and deletion.  See [`KNNDatabase`](@ref), [`stream_knn!`](@ref),
[`knn`](@ref).
"""
module StreamingKNN

using LinearAlgebra: mul!, I, Symmetric, eigen

export KNNDatabase, Metric, InnerProduct, Cosine, Euclidean
export Filter, NoFilter, HeadFilter, pca_basis, head_energy, filter_stats, reset_stats!
export knn, knn_push!, stream_knn!, compact!, recall
export duplicates, representative, isdeleted, nseen, nslots, dimension, metric, deadfraction

include("metrics.jl")
include("topk.jl")
include("storage.jl")
include("filter.jl")
include("database.jl")
include("search.jl")
include("recall.jl")

end # module StreamingKNN
