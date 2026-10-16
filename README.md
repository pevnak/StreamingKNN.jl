# StreamingKNN.jl

Exact k-nearest-neighbour search for **streaming vector data**: at every step
return the k nearest neighbours of the incoming vector among everything seen
so far, then insert it.  Designed for millions of 512–4096-dimensional
embeddings stored as `Float16` on a CPU, with duplicate folding and deletion.

```julia
using StreamingKNN

db = KNNDatabase(1024; metric = InnerProduct(), k = 10)   # empty, Float16 storage
db = KNNDatabase(X0; metric = Cosine())                   # initialised with the columns of X0

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

Extrapolations for 5 M vectors at 1024 dims on the same machine: a single
query takes about 0.7 s, a batched query about 30 ms, and the full causal
stream about one day (`n²` scaling).  A GPU or a 32-core AVX-512 machine
changes these numbers by a large factor.  Run `julia --project -t auto
bench/bench.jl [dim] [n]` to measure your own hardware.

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
julia --project -t auto bench/groundtruth.jl --k 100            # exact results of the whole stream
julia --project -t auto bench/bench.jl 1024 100000              # synthetic kernel micro-benchmarks
```

`groundtruth.jl` stores the exact causal neighbours (ids, scores, and the
representative id of every column) in `bench/groundtruth/`, also not
committed.  An approximate method is evaluated against it with
`recall(gt.ids, pred; k)`.
