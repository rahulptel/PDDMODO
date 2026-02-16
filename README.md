# cuMODD

`cuMODD` is a C++ project for multi-objective optimization using decision diagrams, with CPU and optional CUDA execution paths.

## Build

From the project root:

```bash
make cpu NUM_OBJS=3
```

Optional CUDA build (`nvcc >= 12`):

```bash
make gpu NUM_OBJS=3
```

Build both variants:

```bash
make both NUM_OBJS=3
```

Clean build artifacts:

```bash
make clean
```

## Run

Executable names follow:

- `multiobj_cpu_nobjs<NUM_OBJS>`
- `multiobj_gpu_nobjs<NUM_OBJS>`

CLI:

```bash
./multiobj_cpu_nobjs3 <input-file> <problem-type> <preprocess> <method> <appr-S> <appr-T> <dominance>
```

GPU CLI:

```bash
./multiobj_gpu_nobjs3 <input-file> <problem-type> <preprocess> <method> <appr-S> <appr-T> <dominance>
```

In GPU builds:

- `method=1` uses CUDA top-down enumeration.
- `method=3` uses CUDA dynamic-cutset coupled enumeration.
- `method=2` remains CPU.

For CUDA `method=3`:

- `MULTIOBJ_VERIFY_GPU_METHOD3=1` enables an exact GPU-vs-CPU parity check (point-set and coupling-layer match).
- `MULTIOBJ_GPU_COUPLE_MAX_PAIRS=<int>` overrides max pair count per GPU coupling batch.
- If `dominance>0`, CUDA method-3 currently warns and proceeds without state-dominance pruning.

## CUDA Method-3 Status (Coupled Enumeration)

Current state (as of 2026-02-16):

- CUDA coupled enumeration (`method=3`) is functional and parity-checks exactly against CPU when `MULTIOBJ_VERIFY_GPU_METHOD3=1`.
- Main remaining gap: on the benchmark below, `method=3` is still slower than CUDA top-down (`method=1`).

Benchmark used for tracking:

```bash
./multiobj_gpu_nobjs3 ./data/3/benchmark/SetII/1DKP/KP_p-3_n-80_ins-1.lp-bdd.dat 1 0 3 0 0 0
./multiobj_gpu_nobjs3 ./data/3/benchmark/SetII/1DKP/KP_p-3_n-80_ins-1.lp-bdd.dat 1 0 1 0 0 0
```

Measured `pareto` time summary (seconds, approximate ranges):

- `method=3` (current): `3.10` to `3.25`
- `method=1` (reference): `1.55` to `1.77`

Optimization log (what has already been tried):

- Tree-reduction metadata moved fully to GPU (removed host round-trip in reduction): no clear runtime gain on this instance.
- Archive-style final merge (single running archive, CPU-style): significantly slower (`~7.6` to `8.0` for `method=3`), reverted.
- Reused temporary device buffers in coupling/tree loops (reduced repeated allocations): little/no clear gain.
- Removed per-batch host copy of live counts in coupling (write node sizes on device, copy once): little/no clear gain.
- Reduced explicit sync frequency in hot loops (launch checks + one stream sync per batch/round): little/no clear gain.
- Replaced dominance kernel with tiled shared-memory variant + early reject on first objective: strong improvement (`method=3` moved from about `~4.5` range to `~3.1-3.25` on this benchmark).

## Data

Sample/input data is under `data/`.
