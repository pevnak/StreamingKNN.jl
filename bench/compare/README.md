# Comparison with other Julia nearest-neighbour libraries

Measured on `bench/knntest_1.jls`: the first 20 000 Qwen text embeddings, dimension
4096, stored as `Float16`, of which 8 128 columns repeat an earlier vector exactly.
Metric Euclidean (all vectors have unit norm, so this ranks like cosine and inner
product).  8-core AMD Threadripper 1900X, 8 Julia threads, 8 BLAS threads, `k = 10`,
batch 1024.  Reproduce with `compare.jl` in this directory.

## Workload

Every method runs the **same loop**: a batch is first queried against everything
inserted before it, then inserted.  This is the only streaming contract all the
libraries can honour.  `stream_knn!` additionally answers each column against the
earlier columns *of its own batch*, which is strictly more work and a stricter
result; it is listed separately and is not comparable on the recall column.

Recall is computed after mapping every id to the first occurrence of its vector.
Without that step the exact libraries score 0.79 instead of 1.0, because they report
whichever duplicate copy their index happens to hold.

## Results

| method                        | seconds | vs ours  | recall@10 | kind                              |
|-------------------------------|---------|----------|-----------|-----------------------------------|
| **StreamingKNN** (this package)| **7.1** | 1.00×    | 1.0000    | exact, reference for the table    |
| StreamingKNN, per-column causal| 4.6     | 0.64×    | stricter  | exact, sees within-batch vectors  |
| StreamingKNN + `HeadFilter`   | 20.4    | 2.87×    | 1.0000    | exact, 12 s of it is the PCA fit  |
| SimilaritySearch graph        | 28.5    | 4.01×    | 0.9683    | approximate                       |
| HNSW.jl                       | 31.6    | 4.44×    | 0.9457    | approximate                       |
| Faiss `Flat` (via Python)     | 84.4    | 11.9×    | 1.0000    | exact                             |
| SimilaritySearch exhaustive   | 106.5   | 15.0×    | 1.0000    | exact                             |
| NearestNeighbors `BruteTree`  | 592.3   | 83.2×    | 1.0000    | exact, rebuilt each batch         |
| NearestNeighbors `KDTree`     | 1468.5  | 206×     | 1.0000    | exact, rebuilt each batch         |

Every exact method returns the same neighbours, which cross-validates all four
implementations against each other.

## Reading the table

**Exact search here is a matrix multiply.**  The four exact methods differ only in
how they compute the same inner products.  This package and Faiss hand them to a
threaded BLAS `gemm`; SimilaritySearch and NearestNeighbors evaluate distances one
pair at a time.  That is the whole 83× gap to `BruteTree`, not an algorithmic
difference.  Faiss is 12× slower than this package only because of its per-batch
Python round trip and `Float32` conversion of data this package keeps in `Float16`;
its own kernel is excellent.

**Trees do not work at this dimension.**  `KDTree` is 2.5× *slower* than brute force:
at 4096 dimensions no branch can be pruned, so it pays full traversal overhead on top
of a full scan.

**Approximate indices do not pay off yet at 20 000 vectors.**  Both graph libraries
are ~4× slower than the exact scan while losing 3–5 % of the neighbours, because a
graph must still walk hundreds of 4096-dimensional distance evaluations per query
without BLAS.  They win at a different scale: cost per query grows as `log n` for them
and as `n` for a scan, so the crossover is somewhere in the millions.  Their appeal in
this workload is also limited by construction cost, which is paid on the streaming
path itself.

**Per-column causality is free.**  `stream_knn!` is *faster* than the batch loop
(4.6 s vs 7.1 s, one pass with a reused workspace) while answering a strictly harder
question.  Verified on 513 sampled queries: its k-th neighbour is strictly closer than
the batch-causal answer for 250 of them, never farther, and it never returns a vector
from the future.  Its 0.84 agreement with the batch-causal reference is that semantic
difference, not an error rate.

**The filter is a large-`n` optimisation.**  At 20 000 vectors the one-off PCA fit
(12 s) dominates and the filter loses.  It wins from roughly 10⁵ vectors on: 3.7× at
400 k and 4.8× over the full 1.04 M stream (`bench/perflog.csv`).

## Capabilities

Determined by attempting each operation, not by reading documentation.

| package          | incremental insert | delete | `Float16` | inner product | notes                                   |
|------------------|--------------------|--------|-----------|---------------|------------------------------------------|
| **StreamingKNN** | yes                | yes    | yes       | yes           | duplicate folding, per-column causality  |
| SimilaritySearch | yes                | yes    | yes       | via Distances | needs `BlockMatrixDatabase`, not `MatrixDatabase` |
| HNSW.jl          | partial            | no     | yes       | via Distances | only over a preallocated array; cannot add an unseen vector |
| NearestNeighbors | no                 | no     | yes       | no            | index immutable; metric must be a `Metric` |
| Faiss.jl         | yes                | yes    | no        | yes           | Conda install; fails to precompile (method overwrite) |

Only this package folds duplicates, which on this data removes 41 % of the stored
vectors and the corresponding share of the arithmetic.  `HNSW.jl` cannot grow beyond
the array it was constructed with, so it cannot serve an open-ended stream at all.

## When to use something else

* **Recall below 1.0 is acceptable and the database reaches millions of vectors.**
  SimilaritySearch's `SearchGraph` is the mature Julia option and supports insertion
  and deletion.  Measure the crossover on your own data first.
* **A GPU is available.**  Faiss or cuVS will beat any CPU method by more than an
  order of magnitude; the Python round trip stops mattering once batches are large.
* **Low dimension (under ~20).**  `NearestNeighbors.KDTree` is the right tool there,
  and this package is not.
