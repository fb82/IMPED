# test_merge

Benchmark harness for comparing COLMAP DB merge performance between:
- `legacy-like` behavior (old exhaustive pair traversal + old update pattern)
- `optimized` behavior (current implementation)

## Run

From repo root:

```bash
PYTHONPATH=src python test_merge/benchmark_merge.py --db path/to/db0.db path/to/db1.db --repeats 3
```

Optional output directory:

```bash
PYTHONPATH=src python test_merge/benchmark_merge.py --db path/to/db0.db path/to/db1.db --repeats 3 --out-dir test_merge/out
```

## Output

The script prints per-run timings and a final summary with average time and speedup (`legacy / optimized`).

Note: `legacy-like` mode is intentionally slower to emulate the previous behavior, primarily to quantify the impact of the optimization.
