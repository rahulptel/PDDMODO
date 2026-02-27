# cuMODD

`cuMODD` is a C++ project for multi-objective optimization using decision diagrams, with CPU and optional CUDA execution paths.

## Build

From the project root:

```bash
make NUM_OBJS=3
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
./multiobj_nobjs3 <input-file> <problem-type> <preprocess> <method> <appr-S> <appr-T> <dominance> [backend] [kernel-version]
```

`backend` is optional and can be:
- `cpu` (default when omitted): use CPU pareto frontier enumeration.
- `cuda`: force CUDA pareto frontier enumeration.

`kernel-version` is only used when `backend=cuda`:
- `1`: one block per node.
- `2`: fixed number of blocks per node (2D grid).
- `3`: dynamic number of blocks per node (1D grid + binary-search destination lookup).

If `backend=cuda` and `kernel-version` is omitted, defaults are:
- knapsack (`problem-type=1`) -> `1`
- set packing (`problem-type=2`) -> `2`
- set covering (`problem-type=3`) -> `1`
- tsp (`problem-type=4`) -> `3`

When `backend=cuda`, execution fails fast with a nonzero exit code if CUDA is unavailable or if the selected problem/method combination has no CUDA implementation.

## Data

Sample/input data is under `data/`.
