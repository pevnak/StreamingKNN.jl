# Comparison with other Julia nearest-neighbour libraries

Measured on `bench/knntest_1.jls`: Qwen text embeddings, dimension 4096, stored as
`Float16`.  Metric Euclidean (all vectors have unit norm, so this ranks like cosine
and inner product).  8-core AMD Threadripper 1900X, 8 Julia threads, 8 BLAS threads,
`k = 10`, batch 1024.

```sh
julia --project=benchmark/comparisons -e 'using Pkg; Pkg.develop(path="."); \
    Pkg.add(["NearestNeighbors","SimilaritySearch","HNSW","Distances","Faiss","Serialization"])'
julia --project=benchmark/comparisons -t auto benchmark/comparisons/compare.jl 20000 1024
julia --project=benchmark/comparisons -t auto benchmark/comparisons/compare.jl 100000 1024 \
      "NearestNeighbors brute,NearestNeighbors KDTree"
```

## Workload

Every method runs the **same loop**: a batch is first queried against everything
inserted before it, then inserted.  This is the only streaming contract all the
libraries can honour.  `stream_knn!` additionally answers each column against the
earlier columns *of its own batch*, which is strictly more work and a stricter
result; it is listed separately and its recall column is not comparable.

Recall is computed after mapping every id to the first occurrence of its vector.
Without that step the exact libraries score 0.79 instead of 1.0, because they report
whichever duplicate copy their index happens to hold.

## Results

**n = 100 000** (53 232 columns repeat an earlier vector, leaving 46 768 unique)

| method                        | seconds   | vs ours | recall@10 | kind                             |
|-------------------------------|-----------|---------|-----------|----------------------------------|
| StreamingKNN + `HeadFilter`   | **49.4**  | 0.74×   | 1.0000    | exact, incl. one-off PCA fit     |
| StreamingKNN per-column causal| 61.7      | 0.92×   | stricter  | exact, sees within-batch vectors |
| **StreamingKNN**              | **66.9**  | 1.00×   | 1.0000    | exact, reference for the table   |
| SimilaritySearch graph        | 194.1     | 2.90×   | 0.9616    | approximate                      |
| HNSW.jl                       | 219.4     | 3.28×   | 0.8908    | approximate                      |
| Faiss `Flat` (via Python)     | 1019.8    | 15.2×   | 1.0000    | exact                            |
| SimilaritySearch exhaustive   | 2673.4    | 40.0×   | 1.0000    | exact                            |
| NearestNeighbors `BruteTree`  | ~14 800 † | ~221× † | 1.0000    | exact, rebuilt each batch        |
| NearestNeighbors `KDTree`     | ~36 700 † | ~549× † | 1.0000    | exact, rebuilt each batch        |

† Not run at this size: both scan without BLAS at a cost of `n²`, so they would need
4 and 10 hours.  Extrapolated from the measured 20 000-vector timings below.

**n = 20 000** (8 128 duplicate columns, 11 872 unique)

| method                        | seconds | vs ours | recall@10 |
|-------------------------------|---------|---------|-----------|
| StreamingKNN per-column causal| 4.6     | 0.64×   | stricter  |
| **StreamingKNN**              | **7.1** | 1.00×   | 1.0000    |
| StreamingKNN + `HeadFilter`   | 20.4    | 2.87×   | 1.0000    |
| SimilaritySearch graph        | 28.5    | 4.01×   | 0.9683    |
| HNSW.jl                       | 31.6    | 4.44×   | 0.9457    |
| Faiss `Flat` (via Python)     | 84.4    | 11.9×   | 1.0000    |
| SimilaritySearch exhaustive   | 106.5   | 15.0×   | 1.0000    |
| NearestNeighbors `BruteTree`  | 592.3   | 83.2×   | 1.0000    |
| NearestNeighbors `KDTree`     | 1468.5  | 206×    | 1.0000    |

Every exact method returns the same neighbours, which cross-validates four
independent implementations against each other.

## What changes between the two scales

**The filter crosses over.**  At 20 000 vectors the one-off PCA fit dominates and the
filter loses (2.87×); at 100 000 it is already the fastest method in the table
(0.74×, i.e. 26 % faster than the plain scan, fit included), and it reaches 3.7× at
400 k and 4.8× over the full 1.04 M stream (`bench/perflog.csv`).  Turn it on from
about 10⁵ vectors.

**Approximate indices close the gap but do not overtake.**  SimilaritySearch's graph
goes from 4.01× to 2.90× and HNSW from 4.44× to 3.28× as `n` grows, because a graph
costs `log n` per query where a scan costs `n`.  Extrapolating that trend, the
crossover against the plain scan sits in the low millions, and the filter pushes it
further out.  Neither reaches it at 100 000.

**Approximate recall decays with `n` at fixed parameters.**  HNSW falls from 0.9457
to 0.8908 between the two scales with `M = 16, ef = 50` unchanged.  Holding recall
constant as the database grows means raising `ef`, which costs time the table above
does not charge it.

**Exact non-BLAS methods fall further behind.**  SimilaritySearch's exhaustive index
goes from 15× to 40×, Faiss from 11.9× to 15.2×.  Our scan gets *more* efficient with
scale (larger matrix products, and duplicate folding removes 41 % of vectors at 20 k
but 53 % at 100 k), while a per-pair distance loop does not.

## Reading the table

**Exact search here is a matrix multiply.**  The exact methods differ only in how they
compute the same inner products.  This package and Faiss hand them to a threaded BLAS
`gemm`; SimilaritySearch and NearestNeighbors evaluate distances one pair at a time.
That is the whole gap to `BruteTree`, not an algorithmic difference.  Faiss is slower
than this package only because of its per-batch Python round trip, its `Float32`
conversion of data we keep in `Float16`, and its lack of duplicate folding; its own
kernel is excellent.

**Trees do not work at this dimension.**  `KDTree` is 2.5× *slower* than brute force:
at 4096 dimensions no branch can be pruned, so it pays full traversal overhead on top
of a full scan.

**Per-column causality is free.**  `stream_knn!` is *faster* than the batch loop at
both scales while answering a strictly harder question.  Verified on 513 sampled
queries at n = 20 000: its k-th neighbour is strictly closer than the batch-causal
answer for 250 of them, never farther, and it never returns a vector from the future.
Its recall figure is that semantic difference, not an error rate.

## Capabilities

Determined by attempting each operation, not by reading documentation.

| package          | incremental insert | delete | `Float16` | inner product | notes                                   |
|------------------|--------------------|--------|-----------|---------------|------------------------------------------|
| **StreamingKNN** | yes                | yes    | yes       | yes           | duplicate folding, per-column causality  |
| SimilaritySearch | yes                | yes    | yes       | via Distances | needs `BlockMatrixDatabase`, not `MatrixDatabase` |
| HNSW.jl          | partial            | no     | yes       | via Distances | only over a preallocated array; cannot add an unseen vector |
| NearestNeighbors | no                 | no     | yes       | no            | index immutable; metric must be a `Metric` |
| Faiss.jl         | yes                | yes    | no        | yes           | Conda install; fails to precompile (method overwrite) |

Only this package folds duplicates, which on the 100 000-vector prefix removes 53 % of
the stored vectors and the corresponding share of the arithmetic.  `HNSW.jl` cannot
grow beyond the array it was constructed with, so it cannot serve an open-ended stream
at all.

## When to use something else

* **Recall below 1.0 is acceptable and the database reaches millions of vectors.**
  SimilaritySearch's `SearchGraph` is the mature Julia option and supports insertion
  and deletion.  Measure the crossover on your own data; at 100 000 it has not arrived.
* **A GPU is available.**  Faiss or cuVS will beat any CPU method by more than an order
  of magnitude; the Python round trip stops mattering once batches are large.
* **Low dimension (under ~20).**  `NearestNeighbors.KDTree` is the right tool there,
  and this package is not.
