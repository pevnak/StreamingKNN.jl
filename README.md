# StreamingKNN.jl

Exact k-nearest-neighbour search for **streaming vector data**: at every step
return the k nearest neighbours of the incoming vector among everything seen
so far, then insert it.  Built for millions of 512–4096-dimensional embeddings
stored as `Float16` on a CPU, with duplicate folding and deletion.

## Quick start

```julia
using StreamingKNN

db = KNNDatabase(4096; metric = InnerProduct(), k = 10)   # empty, Float16 storage

# Causal stream: column t is answered against everything before it, then inserted.
ids, scores = stream_knn!(db, X; batchsize = 4096)        # both k × size(X, 2)
```

`ids[r, t]` is the id of the r-th nearest neighbour of `X[:, t]`, best first;
`scores[r, t]` is its score.  Rows past the number of available vectors hold
id `0` and score `NaN`.

## Creating a database

```julia
KNNDatabase(dim; metric, k, filter, chunksize)      # empty, Float16 storage
KNNDatabase{Float32}(dim; ...)                      # other storage type
KNNDatabase(X0; metric = Cosine())                  # initialised with the columns of X0
```

| keyword     | default           | meaning                                        |
|-------------|-------------------|------------------------------------------------|
| `metric`    | `InnerProduct()`  | also `Cosine()` and `Euclidean()`              |
| `k`         | `10`              | default number of neighbours                   |
| `filter`    | `NoFilter()`      | see [Going faster](#going-faster)              |
| `chunksize` | `2^15`            | vectors per storage block                      |

Scores are inner products, cosine similarities, or Euclidean distances
according to the metric, always sorted best first.

## Warm start

A warm start loads a large initial set that you do **not** need neighbours for,
and then hands the database over to the online loop.  Use `append!`, or the
matrix constructor, which calls it:

```julia
db = KNNDatabase(4096; metric = InnerProduct(), k = 10)
append!(db, X_initial)                        # no searching: 1 M vectors in ~6 s
ids, scores = stream_knn!(db, X_online)       # then stream as usual
```

This is cheap because it skips the search.  Inserting `n` vectors costs `O(nd)`;
*streaming* the same `n` costs `O(n²d)`, because every column is also queried
against everything before it.  On 1.04 M of the embeddings described under
[Performance](#performance) that is 5.8 seconds against 99 minutes.

The database that results is **identical** to one built by inserting the same
vectors one at a time, duplicate folding included, so a warm start changes
nothing about what the online phase then returns.

### Loading in pieces

`append!` can be called repeatedly, which is how to load a set that does not fit
in memory all at once.  The result is the same as one big call:

```julia
for part in eachpart(...)                     # however your data is chunked
    append!(db, part)
end
```

Ids are assigned in insertion order across calls, so the id of a vector is its
position in the overall stream, exactly as with `push!`.

### With a filter

A filter is fixed when the database is constructed, and every vector gets its
head computed as it is inserted.  So fit the basis **before** creating the
database, and the warm-start vectors are covered in the same pass:

```julia
basis = pca_basis(X_initial, 1024)            # one-off, ~15 s on 50 k samples
db = KNNDatabase(4096; filter = HeadFilter(basis; gemm = 512))
append!(db, X_initial)
```

There is no way to add a filter to a database that already holds vectors.  If
you expect the database to grow into the hundreds of thousands, decide on the
filter at this point; see [Going faster](#going-faster).

### Cost

Measured on 1 042 898 embeddings of dimension 4096, of which 516 128 are
distinct, 8 threads:

| warm start of 1.04 M vectors | time   | per vector |
|------------------------------|--------|------------|
| no filter                    | 5.8 s  | 5.5 µs     |
| with `HeadFilter`            | 23.6 s | 22.6 µs    |

Hashing (for duplicate detection) and the norms are computed on all threads;
the filter's extra cost is the projection of every distinct vector onto the
basis, which is the work that makes later queries ~5× cheaper.

`append!` works in blocks of `BULK_BLOCK = 16384` columns, so its scratch space
is a fixed ~0.3 GB rather than growing with the input.  Pass `block` to change
it; smaller trades a little speed for less memory.

Memory of the loaded database, per **distinct** vector, plus 8 bytes per id:

```
2 · dim      bytes   Float16 storage
2 · ndims    bytes   filter head, if any
~40          bytes   norms and bookkeeping
```

So 5 M vectors of dimension 4096 with a 1024-wide head need about 51 GB if all
are distinct, or about 26 GB at the 50 % duplicate rate of the benchmark data.

### What you do not get

No neighbours are returned for the warm-start vectors themselves.  If you need
the causal neighbours of the initial set as well, that is the ordinary
quadratic job: stream it with `stream_knn!` instead of appending it.

## Operations

```julia
ids, scores = stream_knn!(db, X; k, batchsize)   # query each column, then insert it
ids, scores = knn_push!(db, x; k)                # one streaming step
ids, scores = knn(db, x; k)                      # query only (also takes a matrix)
id          = push!(db, x)                       # insert only, returns the id
ids         = append!(db, X)                     # insert many (warm start, see above)
delete!(db, id)                                  # remove one id
compact!(db)                                     # reclaim storage of deleted vectors
```

Query a matrix with `knn(db, X; k, batchsize)`; batching amortises the pass
over the data, so prefer it over one column at a time.

## Ids, duplicates and deletion

Every insertion assigns the next **id**, its position in the stream.  Ids are
stable forever.  Vectors that are bitwise identical after conversion to the
storage type share one slot, so duplicates cost no storage and no compute.
Results report the smallest alive id of a matched vector.

```julia
duplicates(db, id)      # all alive ids with the same vector
representative(db, id)  # the one that search results report
isdeleted(db, id)
length(db)              # alive unique vectors
nseen(db)               # ids assigned so far, duplicates included
deadfraction(db)        # share of slots whose ids are all deleted
```

Deleting an id removes it from its slot; the vector leaves the results when
its last id goes.  Deleted slots still take part in the matrix products until
`compact!(db)` removes them, so call it when `deadfraction(db)` grows large.
Ids survive compaction unchanged.

## Going faster

`HeadFilter` prunes candidates with a provable bound instead of scoring them,
and returns **identical results**.  It is opt-in and gives about 4.8× on
4096-dimensional text embeddings:

```julia
basis = pca_basis(X0, 1024)                               # learn once, on initial data
db = KNNDatabase(4096; filter = HeadFilter(basis; gemm = 512))
```

Each vector stores a 1024-coordinate projection (the *head*) plus residual
norms.  The search matrix product runs on the first `gemm = 512` of them, a
Cauchy–Schwarz bound discards candidates that cannot beat the current k-th
score, survivors are bounded again with the rest of the head, and only what
remains is scored exactly from the full vector.

Pruning power depends on the data; check it before committing:

```julia
head_energy(X, 1024; basis)   # fraction of squared norm the head captures; >0.9 is good
filter_stats(db)              # pairs scanned, bounded twice, and scored exactly
```

`HeadFilter(ndims)` without a basis uses the leading coordinates directly, for
Matryoshka-style embeddings whose energy sits in the first coordinates.  Cost:
`2 · ndims` bytes per unique vector, 25 % over the Float16 data for 1024.

## Evaluating an approximate method

```julia
recall(truth_ids, pred_ids; k)
recall(truth_ids, truth_scores, pred_ids, pred_scores; k, metric, atol)
```

The second form also counts a prediction whose exact score is within `atol` of
the k-th true score, which matters when the data contains near-duplicates that
tie within floating-point noise.

## Performance

Exact search touches every stored vector, so a causal stream of `n` vectors
costs `n²d` operations whatever the implementation.  On an 8-core AMD
Threadripper 1900X (AVX2) with 8 threads, over 1.04 M Qwen embeddings of
dimension 4096 (50 % exact duplicates):

| run                                     | time   | equivalent GFLOP/s |
|-----------------------------------------|--------|--------------------|
| full stream, no filter                  | 99 min | 377                |
| full stream, `HeadFilter(1024; gemm=512)` | 21 min | 1810             |

Start Julia with `-t auto`: the conversion, bound and refinement loops use
Julia threads, the matrix products use BLAS threads.  For memory see
[Warm start → Cost](#cost); searching adds a Float32 scratch copy of one
chunk (`4 · dim · chunksize`) and the score buffer
(`4 · chunksize · batchsize`).

## Benchmarks

`benchmark/` expects a serialized `Matrix{Float16}`, one vector per column (not
committed), and logs every run to `benchmark/perflog.csv`.

```sh
julia --project -t auto benchmark/perflog.jl --n 400000 --filter 512:1024
julia --project -t auto benchmark/groundtruth.jl --k 100     # exact results of the whole stream
julia --project -t auto benchmark/bench.jl 1024 100000       # synthetic kernel micro-benchmarks
```

`groundtruth.jl` saves exact neighbours for later recall evaluation; add
`--check-gt <file>` to `perflog.jl` to verify a run against them.

`benchmark/comparisons/` holds a like-for-like comparison against
NearestNeighbors.jl, SimilaritySearch.jl, HNSW.jl and Faiss on the same
streaming workload; see its [README](benchmark/comparisons/README.md) for the
measured table.
