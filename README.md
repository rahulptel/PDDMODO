# cuMODD

`cuMODD` is a C++ project for multi-objective optimization using decision diagrams, with CPU and optional GPU (CUDA) execution paths.

## Build

From the project root:

```bash
make NUM_OBJS=3
```

This default build is CPU-serial (OpenMP disabled unless explicitly enabled).

CPU build with OpenMP acceleration:

```bash
make NUM_OBJS=3 ENABLE_OPENMP=1
```

CPU-only build (no `nvcc` required):

```bash
make NUM_OBJS=3 ENABLE_CUDA=0
```

Clean build artifacts:

```bash
make clean
```

## Run

Executable name:

- `multiobj_nobjs<NUM_OBJS>`

CLI:

```bash
./multiobj_nobjs3 <input-file> <problem-type> <preprocess> <method> <dominance> [options]
```

Backend selection (optional; defaults to CPU):
- named:
  - `--backend cpu|gpu`
- shorthand:
  - `cpu [num-threads]`
  - `gpu [kernel]`

CPU options:
- `--cpu-threads <N>`: positive integer thread count for CPU enumeration (only when built with `ENABLE_OPENMP=1`).
- if CPU backend is selected and no thread count is provided:
  - OpenMP-enabled build: `OMP_NUM_THREADS` is used when valid; otherwise defaults to `1`.
  - OpenMP-disabled build: always runs serially with `1` thread.
- in OpenMP-disabled builds, explicit CPU thread arguments (`--cpu-threads <N>` or `cpu <N>`) fail with a hard error and instruct to rebuild with `ENABLE_OPENMP=1`.

GPU options:
- `--kernel <K>`: select kernel version (`1`, `2`, or `3`).
- `kernel` is only used when backend is `gpu`.
- token `cuda` is rejected; use `gpu`.

Kernel mapping:
- `1`: one block per node.
- `2`: fixed number of blocks per node (2D grid).
- `3`: dynamic number of blocks per node (1D grid + binary-search destination lookup).

If backend is `gpu` and kernel is omitted, defaults are:
- knapsack (`problem-type=1`) -> `1`
- set packing (`problem-type=2`) -> `2`
- set covering (`problem-type=3`) -> `1`
- tsp (`problem-type=4`) -> `3`

Frontier saving options:
- `--save-frontier`: save frontier as gzip-compressed CSV to `<input_stem>.frontier.csv.gz` in the current working directory.
- `--frontier-out <path>`: save frontier as gzip-compressed CSV to the explicit path provided.
- If both are passed, `--frontier-out <path>` is used.
- Optional arguments can be provided in any order.

Stats saving options:
- `--save-stats`: write one JSONL record with run statistics.
- `--stats-out <path>`: write JSONL stats to the explicit path provided.
- If `--save-stats` is passed without `--stats-out`, the default is `<input_stem>.stats.jsonl`.
- `--stats-out <path>` implies `--save-stats`.
- JSONL write mode is overwrite (`trunc`): only the latest run is kept in the target file.
- If stats writing fails, a warning is printed to `stderr` and the run still succeeds.

Performance logging:
- `--perf-log`: emit aggregated phase timings and counters to `stderr` (stdout format remains unchanged).
- Logged wall-clock and CPU-time values are both reported.
- Enumeration timing excludes final lexicographic sorting (sorting is treated as post-processing).

Stdout format (always 3 lines):
- line 1: number of Pareto solutions.
- line 2: CPU total time (`compile_cpu_s + enum_cpu_s`) for backward compatibility.
- line 3: tab-separated stats with existing fields unchanged in order, followed by appended wall-time fields:
  - `compile_wall_s`
  - `enum_wall_s` (excludes final lexicographic sort)
  - `total_wall_s_end_to_end` (includes post-processing such as sort and optional frontier save; measured from run start to stdout reporting)

JSONL schema notes (`--save-stats`):
- One flat JSON object is written per run.
- Includes run configuration, frontier size/shape stats, CPU and wall timings, and aggregated CPU phase counters.
- Key timing semantics:
  - `enum_wall_s` excludes final lexicographic sort.
  - `total_wall_s_end_to_end` includes post-processing before stdout reporting.
  - `total_cpu_s = compile_cpu_s + enum_cpu_s` (same CPU semantics as stdout line 2).

When backend is `gpu`, execution fails fast with a nonzero exit code if CUDA is unavailable or if the selected problem/method combination has no GPU implementation.

## Data

Sample/input data is under `data/`.
