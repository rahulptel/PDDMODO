# MODOBenchmark

This repository collects C++ implementations and benchmark instances for exact
multiobjective discrete optimization. The main focus is on two solvers:

- `dd/`: decision-diagram and network-model based code for
  multiobjective discrete optimization.
- `dpa/`: a defining-point objective-space algorithm for
  multiobjective integer programming.

The repository also includes test instances and small conversion scripts so that
the same benchmark families can be used with the different implementations.

## Main Implementations

### `dd/`

`code_dd` is based on:

David Bergman, Merve Bodur, Carlos Cardonha, and Andre A. Cire,
"Network Models for Multiobjective Discrete Optimization,"
INFORMS Journal on Computing 34(2):990-1005, 2022.
https://doi.org/10.1287/ijoc.2021.1066

This implementation builds binary decision diagrams and related network models
for several problem classes, then enumerates nondominated points through
multiobjective shortest-path style dynamic programming on the resulting graph.
The source tree contains problem readers and model constructors for knapsack,
set packing, set covering, portfolio optimization, absolute-value instances,
and TSP.

The executable is `multiobj`. Its number of objectives is compiled into the
binary with the `NUM_OBJS` make variable.

### `dpa/`

`code_dpa` is based on:

Kerstin Dächert, Tino Fleuren, and Kathrin Klamroth,
"A simple, efficient and versatile objective space algorithm for multiobjective
integer programming," Mathematical Methods of Operations Research 100:351-384,
2024.
https://doi.org/10.1007/s00186-023-00841-0

This implementation reads multiobjective integer programming instances in CPLEX
`.lp` format and computes nondominated objective vectors using a defining-point
objective-space method. The algorithm imports the model with CPLEX/Concert,
extracts the objective rows encoded in the LP, initializes objective-space
boxes, and iteratively refines them through e-constraint scalarizations.

The executable is `main`. By default it uses the two-stage e-constraint variant;
`-a` enables the augmented e-constraint variant and `-v` enables verbose output.

## Repository Layout

- `dd/`: decision-diagram / network-model implementation.
- `dpa/`: defining-point algorithm implementation.

This checkout currently retains the DD and DPA baseline code. Broader upstream
benchmark assets such as additional reference implementations, format
conversion utilities, and `TestInstances/` are not present under this directory
unless added separately.

## Dependencies

Both primary implementations are C++ projects and expect a local IBM ILOG CPLEX
Studio installation.

- `dpa/` depends on CPLEX and Concert.
- `dd/` depends on CPLEX, Concert, CP Optimizer headers/libraries, and
  Boost headers. Its makefile is also prepared for Gurobi, but the Gurobi lines
  are commented out by default.

The supplied makefiles assume CPLEX Studio 12.10 under
`/opt/ibm/ILOG/CPLEX_Studio1210`. Adjust the makefiles or pass variables as
needed for your local installation.

## Building

Build `dpa/`:

```sh
cd dpa
make
```

To override the CPLEX installation path:

```sh
make CPLEX_DIRECTORY=/path/to/CPLEX_Studio
```

Build `dd/` for a fixed number of objectives:

```sh
cd dd
make NUM_OBJS=3
```

The helper script in `dd/compile_all.sh` rebuilds several
objective-count variants. Review it before use because it removes and recreates
local `multiobj` binaries.

## Running

Run `dpa/` on a `.lp` instance:

```sh
cd dpa
./main <instance.lp>
./main <instance.lp> -v
./main <instance.lp> -a
```

`dpa/` writes the nondominated objective vectors to a `.sol` file next to
the input instance.

Run `dd/` on a `.dat` instance:

```sh
cd dd
./multiobj <input-file> <problem-type> <preprocess> <method> <approx-S> <approx-T> <dominance> <n-vars> [variable-order...]
```

Problem types are:

- `1`: knapsack
- `2`: set packing
- `3`: set covering
- `4`: portfolio optimization
- `5`: absolute value
- `6`: TSP

Methods are:

- `1`: top-down BFS
- `2`: bottom-up BFS
- `3`: dynamic layer cutset

For example, after compiling with `NUM_OBJS=3`, a TSP run can be started with:

```sh
./multiobj <instance-dd.dat> 6 0 1 0 0 1 0
```

## Instance Formats

The two primary solvers use different input formats.

- `dd/` uses solver-specific `.dat` files such as `*-dd.dat`.
- `dpa/` uses CPLEX `.lp` files such as `*-dpa.lp`.

For `dpa/`, objective functions are encoded as the final rows of the LP
model. For minimization instances, the upper bound of the final objective row
stores the number of objectives. For maximization instances, the lower bound is
used and signs are adjusted internally.

## Notes

- Generated binaries and object files are intentionally ignored by git.
- Benchmark instances and example data should remain versioned unless a task is
  explicitly about regenerating them.
- Because the build depends on commercial solver installations, CI or automated
  tests should account for machines where CPLEX is not available.
