# StreamingKNN.jl

Exact k-nearest-neighbour search for **streaming vector data**: at every step
return the k nearest neighbours of the incoming vector among everything seen
so far, then insert it.  Designed for millions of 512–4096-dimensional
embeddings stored as `Float16` on a CPU, with duplicate folding and deletion.

```julia
using StreamingKNN

db = KNNDatabase(1024; metric = InnerProduct(), k = 10)   # empty, Float16 storage
db = KNNDatabase(X0; metric = Cosine())                   # initialised with the columns of X0

# opt-in exact filter: 3-4x faster on high-dimensional embeddings, identical results
basis = pca_basis(X0, 1024)                               # learn once on initial data
db = KNNDatabase(4096; filter = HeadFilter(basis; gemm = 512))

# offline stream with causality: result column t holds the neighbours of X[:, t]
# among the initial content and X[:, 1:t-1]; X[:, t] is then inserted.
ids, scores = stream_knn!(db, X; batchsize = 1024)         # k × size(X, 2)

ids, scores = knn_push!(db, x)        # one streaming step (query, then insert)
ids, scores = knn(db, x)              # query only
id          = push!(db, x)            # insert only; returns the stream id
delete!(db, id)                       # remove one id
compact!(db)                          # reclaim storage of deleted vectors
duplicates(db, id)                    # all alive ids with the same vector
```

Scores are inner products, cosine similarities or Euclidean distances
depending on the metric; results are sorted best first.  Rows past the number
of available vectors hold id `0` and score `NaN`.

## Ids, slots and duplicates

Every insertion assigns the next **id** (its position in the stream).  Ids
are stable forever.  Vectors that are `isequal` after conversion to the
storage type share one **slot**, so a duplicate costs no storage or compute.
Results report the smallest alive id of the matched slot; `duplicates(db, id)`
lists the others.  Deleting an id removes it from its slot; the vector leaves
the search results when its last id is deleted.

## How it works

* **Storage.**  Vectors live in column-major `Matrix{T}` chunks (32 768 columns
  by default).  Appending never copies existing data.
* **One kernel, three metrics.**  Each chunk is converted to the compute type
  (`Float32`) and multiplied with the query batch by BLAS.  Every slot carries
  two coefficients so that `scale * ⟨x, q⟩ + shift` is a "larger is better"
  ranking key for the chosen metric, and deleted slots simply have
  `shift = -Inf`.  A bounded heap per query keeps the top k.
* **Causal batches.**  `stream_knn!` inserts the whole batch first (with
  dedup), scores it against the previous content, and then adds a small
  masked self block so that member `t` only sees new unique members before it.
  This is bit-for-bit the same answer as calling `knn_push!` one vector at a
  time, up to floating-point summation order.
* **Small batches.**  For batches up to `direct_batch` (default 16) a direct
  Float16 dot-product kernel is used instead of converting the chunk; it is
  several times faster for single queries.
* **Euclidean refinement.**  The expansion `‖q‖² + ‖x‖² − 2⟨x,q⟩` loses
  precision for tiny distances, so the final k distances are recomputed
  directly.  Reported distances are exact; candidates whose distances differ
  by less than roughly `1e-3` relative to the norms may still swap places.

## Exact filter-and-refine (`HeadFilter`)

Full scans cost `n² d` no matter how they are organised, but most of those
inner products can be *bounded* instead of computed.  `HeadFilter` stores
for every vector a *head*: its projection onto an orthonormal basis `B`
(`dim × ndims`, from `pca_basis`), as Float16, plus the norm of what is
left.  The matrix product of the search runs on the leading `gemm` head
coordinates only, and the Cauchy–Schwarz bound

    ⟨q, x⟩ ≤ ⟨Bᵀq, Bᵀx⟩ + ‖q_tail‖ ‖x_tail‖

tells, for every candidate, whether it can still beat the k-th best score
found so far.  Survivors are bounded once more with the remaining head
coordinates (`gemm+1:ndims`, a short partial dot product) and only the few
that still survive are scored exactly from the full stored vector.  Every
step is a valid upper bound, so the results are the same as without the
filter up to floating-point summation order; only the speed depends on the
data.  The bound is loosest early in a scan, when the k-th best is still
weak, so the kernel refines the best few candidates of the first chunk
before trusting it.

Pruning power is a property of the data: `head_energy(X, ndims; basis)`
reports how much of the squared norm the head captures.  On the 4096-dim
Qwen text embeddings used for benchmarking, 512 PCA coordinates hold 91 % of
the energy, the stage-1 bound discards about 97.5 % of the candidates, the
stage-2 bound (coordinates 513–1024) about 90 % of the rest, and 0.2–0.3 %
of all candidates are scored exactly.  Memory grows by `2 · ndims` bytes per
vector (25 % for `ndims = 1024`).  Without a rotation (`HeadFilter(ndims)`
uses the leading coordinates) the energy must already be concentrated in
the first coordinates, as in Matryoshka embeddings; the benchmark file does
not have that property, its energy is spread uniformly.

`filter_stats(db)` returns the counters of the current run (candidate pairs
scanned, bounded a second time, and refined exactly).

## Performance

Exact search must touch every stored vector, so the total cost of a causal
stream of `n` vectors is `n² d` floating-point operations regardless of
implementation.  Measured on an 8-core AMD Threadripper 1900X (AVX2, no
AVX-512), 8 Julia threads, `dim = 1024`, `Float16` storage:

| workload                                   | measured                       |
|--------------------------------------------|--------------------------------|
| `stream_knn!`, 100 k vectors, batch 1024   | 37 s, ~280 GFLOP/s sustained   |
| `stream_knn!`, 100 k vectors, batch 4096   | 34 s, ~305 GFLOP/s sustained   |
| single `knn` over 100 k vectors            | 14 ms                          |
| 1024 batched queries over 100 k vectors    | 0.65 ms per query              |

On the real benchmark data (`bench/knntest_1.jls`, 1 042 898 Qwen text
embeddings of dimension 4096, 50 % exact duplicates), batch 4096, 8 threads,
inner product, see `bench/perflog.csv`:

| run                                           | time    | equivalent GFLOP/s |
|-----------------------------------------------|---------|--------------------|
| full stream, no filter, k = 100 (ground truth)| 99 min  | 377                |
| full stream, `HeadFilter(1024; gemm = 512)`, k = 10 | 21 min | 1810         |
| first 400 k, no filter (extrapolated)         | ~862 s  | 377                |
| first 400 k, `HeadFilter(1024; gemm = 512)`   | 230 s   | 1414               |

"Equivalent GFLOP/s" counts the work a full scan would do, so it measures
the speed-up directly: the filter gives 4.8× on the full stream (3.7× on the
first 400 k; pruning improves as the database grows) with results identical
to the exact scan (recall 1.0 against the stored ground truth up to Float32
ties).  On the full stream 1.7 % of the candidate pairs pass the first
bound and 0.16 % are scored exactly.

Extrapolations for 5 M vectors at 1024 dims on the same machine without the
filter: a single query takes about 0.7 s, a batched query about 30 ms, and
the full causal stream about one day (`n²` scaling).  A GPU or a 32-core
AVX-512 machine changes these numbers by a large factor.  Run
`julia --project -t auto bench/bench.jl [dim] [n]` to measure your own
hardware.

Memory is the Float16 storage (`2 · dim · n` bytes: 10 GB for 5 M × 1024,
40 GB for 5 M × 4096) plus a `Float32` scratch copy of one chunk
(`4 · dim · chunksize`, i.e. 128–512 MB) and the score buffer
(`4 · chunksize · batchsize`).

## Tips

* Start Julia with `-t auto`; the conversion and top-k loops use Julia threads,
  the matrix products use BLAS threads.
* Larger `batchsize` means fewer passes over the data; 1024–4096 is a good range.
* Call `compact!` when `deadfraction(db)` becomes large; deleted slots still
  participate in the matrix products until then.
* `KNNDatabase{Float32}` or `KNNDatabase{Float64}` skip the conversion step
  entirely at the cost of memory.

## Benchmarks, performance log and ground truth

The `bench/` directory expects a real data file (a serialized `Matrix{Float16}`,
one vector per column; not committed) and keeps a running log of how the
engine performs on it in `bench/perflog.csv`, one row per run with the
commit hash, machine, thread counts, timings and sustained GFLOP/s.

```sh
julia --project -t auto bench/perflog.jl --n 100000             # prefix benchmark, appends a row
julia --project -t auto bench/perflog.jl --n 400000 --filter 512:1024 \
      --check-gt bench/groundtruth/knntest_1_ip_k100_n1042898.jls  # with the filter, checked against ground truth
julia --project -t auto bench/groundtruth.jl --k 100            # exact results of the whole stream
julia --project -t auto bench/bench.jl 1024 100000              # synthetic kernel micro-benchmarks
```

`groundtruth.jl` stores the exact causal neighbours (ids, scores, and the
representative id of every column) in `bench/groundtruth/`, also not
committed.  An approximate method is evaluated against it with
`recall(gt.ids, pred; k)`, or, when the data contains near-duplicates whose
scores tie within noise, with the score-tolerant form
`recall(gt.ids, gt.scores, pred_ids, pred_scores; k, metric, atol)`, which
also accepts a prediction whose exact score is within `atol` of the k-th true
score.
