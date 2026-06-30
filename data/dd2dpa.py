#!/usr/bin/env python3
"""Convert cuMODD .dat instances to DPA-readable LP files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ROOTS = tuple(Path(__file__).resolve().parent / str(i) for i in range(3, 8))


class ConversionError(Exception):
    pass


@dataclass(frozen=True)
class LinearRow:
    name: str
    terms: Sequence[tuple[int, str]]
    sense: str
    rhs: int


def read_ints(path: Path) -> list[int]:
    try:
        return [int(token) for token in path.read_text().split()]
    except ValueError as exc:
        raise ConversionError(f"{path}: expected integer tokens") from exc


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ConversionError(f"{path}: {message}")


def take(tokens: list[int], index: int, path: Path, context: str) -> tuple[int, int]:
    if index >= len(tokens):
        raise ConversionError(f"{path}: unexpected end of file while reading {context}")
    return tokens[index], index + 1


def take_many(tokens: list[int], index: int, count: int, path: Path, context: str) -> tuple[list[int], int]:
    if index + count > len(tokens):
        raise ConversionError(f"{path}: unexpected end of file while reading {context}")
    return tokens[index : index + count], index + count


def lp_term_chunks(terms: Sequence[tuple[int, str]], chunk_size: int = 8) -> list[str]:
    pieces: list[str] = []
    for coeff, var in terms:
        if coeff == 0:
            continue
        sign = "+" if coeff > 0 else "-"
        pieces.append(f"{sign} {abs(coeff)} {var}")
    if not pieces:
        return ["0"]
    return [" ".join(pieces[i : i + chunk_size]) for i in range(0, len(pieces), chunk_size)]


def dense_terms(coeffs: Sequence[int], prefix: str = "X") -> list[tuple[int, str]]:
    return [(coeff, f"{prefix}{index}") for index, coeff in enumerate(coeffs, start=1) if coeff != 0]


def arc_var(i: int, j: int) -> str:
    return f"X_{i}_{j}"


def format_lp(
    sense: str,
    rows: Sequence[LinearRow],
    objective_rows: Sequence[LinearRow],
    binary_vars: Sequence[str],
    bounds: Sequence[str] = (),
) -> str:
    dummy_objective = f" obj: 0 {binary_vars[0]}" if binary_vars else " obj: 0"
    lines: list[str] = [sense, dummy_objective, "Subject To"]
    for row in [*rows, *objective_rows]:
        lines.append(f" {row.name}:")
        lines.extend(f"  {chunk}" for chunk in lp_term_chunks(row.terms))
        lines.append(f"  {row.sense} {row.rhs}")
    if bounds:
        lines.append("Bounds")
        lines.extend(f" {bound}" for bound in bounds)
    if binary_vars:
        lines.append("Binary")
        lines.extend(f" {var}" for var in binary_vars)
    lines.append("End")
    lines.append("")
    return "\n".join(lines)


def parse_knapsack(path: Path) -> str:
    tokens = read_ints(path)
    index = 0
    n_vars, index = take(tokens, index, path, "number of variables")
    n_cons, index = take(tokens, index, path, "number of constraints")
    n_objs, index = take(tokens, index, path, "number of objectives")
    require(n_vars > 0 and n_cons >= 0 and n_objs > 0, path, "invalid knapsack header")

    objectives: list[list[int]] = []
    for obj in range(n_objs):
        coeffs, index = take_many(tokens, index, n_vars, path, f"objective {obj + 1}")
        objectives.append(coeffs)

    rows: list[LinearRow] = []
    for cons in range(n_cons):
        coeffs, index = take_many(tokens, index, n_vars, path, f"constraint {cons + 1}")
        rhs, index = take(tokens, index, path, f"rhs {cons + 1}")
        rows.append(LinearRow(f"c{cons + 1}", dense_terms(coeffs), "<=", rhs))

    require(index == len(tokens), path, "trailing tokens after knapsack instance")
    objective_rows = [
        LinearRow(f"obj{obj + 1}", dense_terms(coeffs), ">=", obj + 1)
        for obj, coeffs in enumerate(objectives)
    ]
    binary_vars = [f"X{i}" for i in range(1, n_vars + 1)]
    return format_lp("Maximize", rows, objective_rows, binary_vars)


def parse_binproblem(path: Path) -> str:
    tokens = read_ints(path)
    index = 0
    n_vars, index = take(tokens, index, path, "number of variables")
    n_cons, index = take(tokens, index, path, "number of constraints")
    n_objs, index = take(tokens, index, path, "number of objectives")
    require(n_vars > 0 and n_cons >= 0 and n_objs > 0, path, "invalid binproblem header")

    objectives: list[list[int]] = []
    for obj in range(n_objs):
        coeffs, index = take_many(tokens, index, n_vars, path, f"objective {obj + 1}")
        objectives.append(coeffs)

    rows: list[LinearRow] = []
    for cons in range(n_cons):
        count, index = take(tokens, index, path, f"constraint {cons + 1} size")
        require(count >= 0, path, f"constraint {cons + 1} has negative size")
        vars_in_row, index = take_many(tokens, index, count, path, f"constraint {cons + 1} variables")
        for var in vars_in_row:
            require(1 <= var <= n_vars, path, f"constraint {cons + 1} references variable {var}")
        terms = [(1, f"X{var}") for var in vars_in_row]
        rows.append(LinearRow(f"c{cons + 1}", terms, "<=", 1))

    require(index == len(tokens), path, "trailing tokens after binproblem instance")
    objective_rows = [
        LinearRow(f"obj{obj + 1}", dense_terms(coeffs), ">=", obj + 1)
        for obj, coeffs in enumerate(objectives)
    ]
    binary_vars = [f"X{i}" for i in range(1, n_vars + 1)]
    return format_lp("Maximize", rows, objective_rows, binary_vars)


def parse_tsp(path: Path) -> str:
    tokens = read_ints(path)
    index = 0
    n_objs, index = take(tokens, index, path, "number of objectives")
    n_cities, index = take(tokens, index, path, "number of cities")
    require(n_objs > 0 and n_cities >= 2, path, "invalid TSP header")

    matrices: list[list[list[int]]] = []
    for obj in range(n_objs):
        matrix: list[list[int]] = []
        for row in range(n_cities):
            coeffs, index = take_many(tokens, index, n_cities, path, f"objective {obj + 1} row {row}")
            matrix.append(coeffs)
        matrices.append(matrix)

    require(index == len(tokens), path, "trailing tokens after TSP instance")

    rows: list[LinearRow] = []
    for city in range(n_cities):
        out_terms = [(1, arc_var(city, j)) for j in range(n_cities) if j != city]
        in_terms = [(1, arc_var(i, city)) for i in range(n_cities) if i != city]
        rows.append(LinearRow(f"out_{city}", out_terms, "=", 1))
        rows.append(LinearRow(f"in_{city}", in_terms, "=", 1))

    # Miller-Tucker-Zemlin subtour elimination over non-depot cities.
    for i in range(1, n_cities):
        for j in range(1, n_cities):
            if i == j:
                continue
            terms = [(1, f"U{i}"), (-1, f"U{j}"), (n_cities - 1, arc_var(i, j))]
            rows.append(LinearRow(f"mtz_{i}_{j}", terms, "<=", n_cities - 2))

    objective_rows: list[LinearRow] = []
    for obj, matrix in enumerate(matrices, start=1):
        terms = [
            (matrix[i][j], arc_var(i, j))
            for i in range(n_cities)
            for j in range(n_cities)
            if i != j and matrix[i][j] != 0
        ]
        objective_rows.append(LinearRow(f"obj{obj}", terms, "<=", obj))

    binary_vars = [arc_var(i, j) for i in range(n_cities) for j in range(n_cities) if i != j]
    bounds = [f"1 <= U{i} <= {n_cities - 1}" for i in range(1, n_cities)]
    return format_lp("Minimize", rows, objective_rows, binary_vars, bounds)


def convert_file(path: Path, overwrite: bool = False) -> Path:
    if path.suffix != ".dat":
        raise ConversionError(f"{path}: expected a .dat file")

    family = path.parent.name
    if family == "knapsack":
        content = parse_knapsack(path)
    elif family == "binproblem":
        content = parse_binproblem(path)
    elif family == "tsp":
        content = parse_tsp(path)
    else:
        raise ConversionError(f"{path}: unsupported data family {family!r}")

    out_path = path.with_suffix(".lp")
    if out_path.exists() and not overwrite:
        raise ConversionError(f"{out_path}: output already exists; use --overwrite")
    out_path.write_text(content)
    return out_path


def find_dat_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.dat")))
        elif path.is_file() and path.suffix == ".dat":
            files.append(path)
        else:
            raise ConversionError(f"{path}: expected a .dat file or directory")
    return files


def default_paths() -> list[Path]:
    return [path for path in DEFAULT_ROOTS if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert cuMODD data/{3..7} .dat instances to DPA-readable same-stem .lp files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="optional .dat files or directories to convert; defaults to data/3 through data/7",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing .lp files")
    args = parser.parse_args()

    paths = args.paths or default_paths()
    try:
        files = find_dat_files(paths)
    except ConversionError as error:
        print(error)
        return 2

    converted = 0
    skipped = 0
    for path in files:
        try:
            out_path = convert_file(path, overwrite=args.overwrite)
        except ConversionError as error:
            skipped += 1
            print(f"SKIP: {error}")
            continue
        converted += 1
        print(f"WROTE: {out_path}")

    print(f"converted={converted} skipped={skipped}")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
