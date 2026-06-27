#!/usr/bin/env python3
"""Summarize Pareto frontier enumeration stats across farm outputs.

This script reads table entries from run/cc/<farm>/table.dat, maps RUNn folders to
table row n, parses instance metadata (Problem, N, K) from filenames, and
aggregates timing metrics into CSV and LaTeX outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_FARMS = [
    "cpu-coup-4",
    "cpu-coup-8",
    "cpu-coup-16",
    "gpu-td",
    "gpu-coup",
]

CSV_HEADER = [
    "N",
    "K",
    "Method",
    "Time (S)",
    "Speed-up",
    "S^(M, T)",
]

PROBLEM_ORDER = {"MOKP": 0, "MOSPP": 1, "MOTSP": 2}
PROBLEM_CAPTIONS = {
    "MOKP": "MOKP Pareto Enumeration Runtime Comparison Across TD/Coup GPU Methods",
    "MOSPP": "MOSPP Pareto Enumeration Runtime Comparison Across TD/Coup GPU Methods",
    "MOTSP": "MOTSP Pareto Enumeration Runtime Comparison Across TD/Coup GPU Methods",
}
ALLOWED_CASES_BY_PROBLEM: Dict[str, set[Tuple[int, int]]] = {
    "MOKP": {
        (40, 6), (40, 7),
        (50, 5), (50, 6), (50, 7),
        (60, 4), (60, 5), (60, 6), (60, 7),
        (70, 3), (70, 4), (70, 5), (70, 6), (70, 7),
    },
    "MOSPP": {
        (150, 3), (150, 4), (150, 5), (150, 6), (150, 7),
        (200, 3), (200, 4),
    },
    "MOTSP": {
        (15, 4), (15, 5), (15, 6), (15, 7),
    },
}

RUN_DIR_RE = re.compile(r"^RUN(\d+)$")
FARM_RE = re.compile(r"^(cpu|gpu)-(coup|td)(?:-(\d+))?$", re.IGNORECASE)
METHOD_LABEL_RE = re.compile(r"^(Coup|TD)-(CPU|GPU)(?:-(\d+))?$")

KP_CLASSIC_RE = re.compile(r"^KP_p-(\d+)_n-(\d+)_ins-\d+$")
KP_ALT_RE = re.compile(r"^knapsack-(\d+)-(\d+)-\d+-\d+$")
BP_RE = re.compile(r"^bp-(\d+)-\d+-(\d+)-\d+-\d+$")
TSP_RE = re.compile(r"^tsp-nobj(\d+)-ncities(\d+)-seed\d+$")
CPU_OOM_RE = re.compile(r"std::bad_alloc|cannot allocate memory|out of memory|oom", re.IGNORECASE)
GPU_OOM_RE = re.compile(
    r"cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED|"
    r"thrust::system::system_error|hipErrorOutOfMemory|failed to allocate device memory|"
    r"CUDA out of memory|out of memory",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RowKey:
    problem: str
    n: int
    k: int
    method: str


@dataclass
class Aggregate:
    count: int = 0
    total_wall_sum: float = 0.0
    enum_wall_sum: float = 0.0
    memory_out_count: int = 0
    timeout_count: int = 0

    def add(self, total_wall_s: float, enum_wall_s: float) -> None:
        self.count += 1
        self.total_wall_sum += total_wall_s
        self.enum_wall_sum += enum_wall_s


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Pareto frontier enumeration timings across farms."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current working directory).",
    )
    parser.add_argument(
        "--farms",
        default=",".join(DEFAULT_FARMS),
        help="Comma-separated farm names to process.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Root directory containing outputs/<farm>.",
    )
    parser.add_argument(
        "--cc-root",
        type=Path,
        default=None,
        help="Root directory containing <farm>/table.dat (default: project-root/run/cc).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV output path.",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=None,
        help="LaTeX output path.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Floating-point output precision (default: 6).",
    )
    parser.add_argument(
        "--speedup-baseline",
        default="cpu-coup-4",
        help=(
            "Baseline farm or method label for speed-up ratios "
            "(default: cpu-coup-4)."
        ),
    )
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    if args.outputs_root is None:
        args.outputs_root = args.project_root / "outputs"
    else:
        args.outputs_root = args.outputs_root.resolve()

    if args.cc_root is None:
        args.cc_root = args.project_root / "run" / "cc"
    else:
        args.cc_root = args.cc_root.resolve()

    if args.csv_out is None:
        args.csv_out = args.project_root / "results" / "pareto_summary.csv"
    else:
        args.csv_out = args.csv_out.resolve()

    if args.tex_out is None:
        args.tex_out = args.project_root / "results" / "pareto_summary.tex"
    else:
        args.tex_out = args.tex_out.resolve()

    farms = [farm.strip() for farm in args.farms.split(",") if farm.strip()]
    if not farms:
        parser.error("No farms provided via --farms.")
    args.farms = farms

    if args.precision < 0:
        parser.error("--precision must be non-negative.")
    args.speedup_baseline_method = normalize_speedup_baseline(args.speedup_baseline)
    return args


def normalize_speedup_baseline(baseline: str) -> str:
    method, skip_silently = method_from_farm_and_tokens(baseline, [])
    if method is not None and not skip_silently:
        return method
    return baseline


def parse_problem_n_k(instance_path: str) -> Optional[Tuple[str, int, int]]:
    stem = Path(instance_path).stem

    match = KP_CLASSIC_RE.match(stem)
    if match:
        return ("MOKP", int(match.group(2)), int(match.group(1)))

    match = KP_ALT_RE.match(stem)
    if match:
        return ("MOKP", int(match.group(1)), int(match.group(2)))

    match = BP_RE.match(stem)
    if match:
        return ("MOSPP", int(match.group(1)), int(match.group(2)))

    match = TSP_RE.match(stem)
    if match:
        return ("MOTSP", int(match.group(2)), int(match.group(1)))

    return None


def extract_flag_value(tokens: List[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def method_from_farm_and_tokens(farm: str, tokens: List[str]) -> Tuple[Optional[str], bool]:
    """Return (method_label, skip_silently)."""
    match = FARM_RE.match(farm)
    if not match:
        warn(f"Unsupported farm naming convention: {farm}")
        return None, False

    device = match.group(1).lower()
    enum_name_raw = match.group(2).lower()
    threads_raw = match.group(3)

    if enum_name_raw == "coup":
        enum_name = "Coup"
    else:
        enum_name = "TD"

    if device == "cpu":
        # CPU-K3 rows are intentionally excluded from summaries.
        cpu_kernel = extract_flag_value(tokens, "--cpu-kernel")
        if cpu_kernel is not None and cpu_kernel != "1":
            return None, True

    method = f"{enum_name}-{device.upper()}"
    if device == "cpu":
        threads = int(threads_raw) if threads_raw is not None else 1
        if threads > 1:
            method += f"-{threads}"
    return method, False


def parse_table_for_farm(
    farm: str, table_path: Path
) -> Tuple[Dict[int, RowKey], List[RowKey], set[int]]:
    run_to_key: Dict[int, RowKey] = {}
    expected: List[RowKey] = []
    skipped_run_ids: set[int] = set()

    if not table_path.is_file():
        warn(f"Missing table file for farm {farm}: {table_path}")
        return run_to_key, expected, skipped_run_ids

    with table_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                tokens = shlex.split(line)
            except ValueError as exc:
                warn(f"Failed to parse table line {line_num} in {table_path}: {exc}")
                continue

            if not tokens:
                continue

            run_id = line_num
            if tokens[0].isdigit():
                run_id = int(tokens[0])
                tokens = tokens[1:]

            if len(tokens) < 2:
                warn(f"Malformed table line {line_num} in {table_path}: {line}")
                continue

            instance_path = tokens[1]
            parsed = parse_problem_n_k(instance_path)
            if parsed is None:
                warn(
                    f"Unrecognized instance filename format on line {line_num} in {table_path}: "
                    f"{instance_path}"
                )
                continue

            method, skip_silently = method_from_farm_and_tokens(farm, tokens)
            if method is None:
                if skip_silently:
                    skipped_run_ids.add(run_id)
                continue

            problem, n, k = parsed
            key = RowKey(problem=problem, n=n, k=k, method=method)
            run_to_key[run_id] = key
            expected.append(key)

    return run_to_key, expected, skipped_run_ids


def first_nonempty_jsonl_record(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                return json.loads(stripped)
    except json.JSONDecodeError as exc:
        warn(f"Invalid JSON in stats file {path}: {exc}")
        return None
    except OSError as exc:
        warn(f"Failed to read stats file {path}: {exc}")
        return None
    return None


def read_text_if_exists(path: Path) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        warn(f"Failed reading log file {path}: {exc}")
    return ""


def classify_failure(case_stderr: str, case_stdout: str, is_gpu_case: bool) -> str:
    combined = f"{case_stderr}\n{case_stdout}"
    if CPU_OOM_RE.search(combined):
        return "memory"
    if is_gpu_case and GPU_OOM_RE.search(combined):
        return "memory"
    # single_case.sh marks remaining unsuccessful runs as timeout via exit code 124.
    # Exit code is unavailable in RUN folders, so unresolved failures are counted as timeout.
    return "timeout"


def scan_output_runs(
    farm: str, output_farm_dir: Path, run_to_key: Dict[int, RowKey], skipped_run_ids: set[int]
) -> Iterable[Tuple[RowKey, Optional[dict], str, str, str, bool]]:
    if not output_farm_dir.is_dir():
        warn(f"Missing outputs directory for farm {farm}: {output_farm_dir}")
    else:
        known_run_ids = set(run_to_key) | skipped_run_ids
        for run_dir in sorted(output_farm_dir.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            match = RUN_DIR_RE.match(run_dir.name)
            if not match:
                continue
            run_id = int(match.group(1))
            if run_id not in known_run_ids:
                warn(f"No table mapping found for {run_dir} (farm {farm}, run {run_id})")

    for run_id in sorted(run_to_key):
        key = run_to_key[run_id]
        run_dir = output_farm_dir / f"RUN{run_id}"
        run_name = run_dir.name

        frontier_files = sorted(run_dir.glob("*.frontier.csv.gz")) if run_dir.is_dir() else []
        has_frontier = bool(frontier_files)
        if len(frontier_files) > 1:
            warn(f"Multiple frontier files in {run_dir}; using frontier presence only")

        record: Optional[dict] = None
        if has_frontier:
            stats_files = sorted(run_dir.glob("*.stats.jsonl"))
            if stats_files:
                if len(stats_files) > 1:
                    warn(f"Multiple stats files in {run_dir}; using {stats_files[0].name}")
                record = first_nonempty_jsonl_record(stats_files[0])
            else:
                warn(f"Frontier exists but stats file is missing in {run_dir}")

        case_stderr = read_text_if_exists(run_dir / "case.stderr")
        case_stdout = read_text_if_exists(run_dir / "case.stdout")
        yield key, record, run_name, case_stderr, case_stdout, has_frontier


def extract_metrics_for_key(key: RowKey, record: dict, farm: str, run_hint: str) -> Optional[Tuple[float, float]]:
    status = record.get("status", {}).get("status_state")
    if status != "ok":
        warn(f"Skipping non-ok stats ({status}) for {farm}/{run_hint}")
        return None

    try:
        wall = record["timing"]["wall"]
        total_wall_s = float(wall["wall_compile_s"]) + float(wall["wall_enumeration_s"])
        enum_wall_s = float(wall["wall_enumeration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        warn(f"Malformed stats payload for {farm}/{run_hint}: {exc}")
        return None
    return total_wall_s, enum_wall_s


def format_time_ceil_int(value: float) -> str:
    return str(int(math.ceil(value)))


def format_speedup(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.2f}x"


def format_solved_text(agg: Aggregate) -> str:
    solved = max(0, 10 - agg.memory_out_count - agg.timeout_count)
    if agg.memory_out_count == 0 and agg.timeout_count == 0:
        return str(solved)
    return f"{solved}^({agg.memory_out_count}, {agg.timeout_count})"


def format_solved_tex(agg: Aggregate) -> str:
    solved = max(0, 10 - agg.memory_out_count - agg.timeout_count)
    if agg.memory_out_count == 0 and agg.timeout_count == 0:
        return rf"${solved}$"
    return rf"${solved}^{{({agg.memory_out_count}, {agg.timeout_count})}}$"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def parse_method_label(method: str) -> Optional[Tuple[str, str, Optional[int]]]:
    match = METHOD_LABEL_RE.match(method)
    if not match:
        return None
    enum_name = match.group(1)
    device = match.group(2)
    threads = int(match.group(3)) if match.group(3) is not None else None
    return enum_name, device, threads


def method_sort_key(method: str) -> Tuple[int, int, int, str]:
    parsed = parse_method_label(method)
    if parsed is None:
        return (99, 99, 99, method)

    enum_name, device, threads = parsed
    if device == "GPU":
        sophistication = {"TD": 0, "Coup": 1}.get(enum_name, 99)
        return (10**9, sophistication, 0, method)

    parallelism = 1 if threads is None else threads
    sophistication = {"TD": 0, "Coup": 1}.get(enum_name, 99)
    return (parallelism, sophistication, 0, method)


def method_display_name(problem: str, method: str) -> str:
    parsed = parse_method_label(method)
    if parsed is None:
        return method

    enum_name, device, threads = parsed
    plus = "+" if problem == "MOKP" else ""
    label = f"{enum_name}{plus}"
    if device == "GPU":
        return f"G-{label}"
    if threads is not None:
        return f"{label}-{threads}"
    return method


def tex_method_group_and_order(method: str) -> Tuple[int, int, str]:
    parsed = parse_method_label(method)
    if parsed is None:
        return (99, 99, method)

    enum_name, device, threads = parsed
    if device == "CPU" and enum_name == "TD":
        return (0, 0 if threads is None else threads, method)
    if device == "CPU" and enum_name == "Coup":
        return (1, 0 if threads is None else threads, method)
    if device == "GPU" and enum_name == "TD":
        return (2, 0, method)
    if device == "GPU" and enum_name == "Coup":
        return (2, 1, method)
    return (98, 98, method)


def is_gpu_method(method: str) -> bool:
    parsed = parse_method_label(method)
    if parsed is None:
        return method.endswith("-GPU")
    return parsed[1] == "GPU"


def sort_rows(keys: Iterable[RowKey]) -> List[RowKey]:
    return sorted(
        keys,
        key=lambda key: (
            PROBLEM_ORDER.get(key.problem, 99),
            key.n,
            key.k,
            *method_sort_key(key.method),
        ),
    )


def write_csv(
    csv_path: Path,
    ordered_keys: List[RowKey],
    aggregates: Dict[RowKey, Aggregate],
    precision: int,
    speedup_baseline_method: str,
) -> None:
    baseline_by_size: Dict[Tuple[str, int, int], Optional[float]] = {}
    for key in ordered_keys:
        if key.method != speedup_baseline_method:
            continue
        agg = aggregates.get(key, Aggregate())
        if agg.count > 0:
            baseline_by_size[(key.problem, key.n, key.k)] = agg.total_wall_sum / agg.count

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for key in ordered_keys:
            agg = aggregates.get(key, Aggregate())
            avg_total = ""
            speedup = ""
            if agg.count > 0:
                avg_total_value = agg.total_wall_sum / agg.count
                avg_total = format_time_ceil_int(avg_total_value)
                baseline = baseline_by_size.get((key.problem, key.n, key.k))
                if baseline is not None and avg_total_value > 0.0:
                    speedup = format_speedup(baseline / avg_total_value)
            writer.writerow(
                [
                    key.n,
                    key.k,
                    method_display_name(key.problem, key.method),
                    avg_total,
                    speedup,
                    format_solved_text(agg),
                ]
            )


def write_tex(
    tex_path: Path,
    ordered_keys: List[RowKey],
    aggregates: Dict[RowKey, Aggregate],
    precision: int,
    speedup_baseline_method: str,
) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [r"% Requires: \usepackage{booktabs,graphicx,multirow}"]
    eps = 1e-12
    for problem in ("MOKP", "MOSPP", "MOTSP"):
        allowed_cases = ALLOWED_CASES_BY_PROBLEM.get(problem)
        problem_rows = [
            key
            for key in ordered_keys
            if key.problem == problem
            and (allowed_cases is None or (key.n, key.k) in allowed_cases)
        ]
        if not problem_rows:
            continue

        sizes = sorted({(key.n, key.k) for key in problem_rows})
        methods = sorted({key.method for key in problem_rows}, key=tex_method_group_and_order)
        key_lookup: Dict[Tuple[int, int, str], RowKey] = {
            (key.n, key.k, key.method): key for key in problem_rows
        }

        # Best (minimum) total-time averages across methods for each (N, K).
        best_by_size: Dict[Tuple[int, int], Optional[float]] = {}
        baseline_by_size: Dict[Tuple[int, int], Optional[float]] = {}
        for size in sizes:
            best_total: Optional[float] = None
            baseline_total: Optional[float] = None
            for method in methods:
                key = key_lookup.get((size[0], size[1], method))
                if key is None:
                    continue
                agg = aggregates.get(key, Aggregate())
                if agg.count <= 0:
                    continue
                avg_total = agg.total_wall_sum / agg.count
                if best_total is None or avg_total < best_total:
                    best_total = avg_total
                if method == speedup_baseline_method:
                    baseline_total = avg_total
            best_by_size[size] = best_total
            baseline_by_size[size] = baseline_total

        if len(lines) > 1:
            lines.append("")
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\scriptsize")
        caption = PROBLEM_CAPTIONS.get(problem, problem)
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{rrlrrl|rrlrrl}")
        lines.append(r"\toprule")
        lines.append(
            r"N & K & Method & Time (S) & Speed-up & $S^{(M,T)}$ & "
            r"N & K & Method & Time (S) & Speed-up & $S^{(M,T)}$ \\"
        )
        lines.append(r"\midrule")

        size_pairs: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]] = []
        for idx in range(0, len(sizes), 2):
            left_size = sizes[idx]
            right_size = sizes[idx + 1] if idx + 1 < len(sizes) else None
            size_pairs.append((left_size, right_size))

        def render_cells(
            size: Optional[Tuple[int, int]], method: str, method_idx: int
        ) -> Tuple[str, str, str, str, str, str]:
            if size is None:
                return ("", "", "", "", "", "")

            n, k = size
            n_text = rf"\multirow{{{len(methods)}}}{{*}}{{{n}}}" if method_idx == 0 else ""
            k_text = rf"\multirow{{{len(methods)}}}{{*}}{{{k}}}" if method_idx == 0 else ""
            method_text = latex_escape(method_display_name(problem, method))
            key = key_lookup.get((n, k, method))
            if key is None:
                return (n_text, k_text, method_text, "--", "--", r"$10$")

            agg = aggregates.get(key, Aggregate())
            if agg.count > 0:
                avg_total_value = agg.total_wall_sum / agg.count
                avg_total = format_time_ceil_int(avg_total_value)
                baseline = baseline_by_size.get(size)
                speedup = "--"
                if baseline is not None and avg_total_value > 0.0:
                    speedup = format_speedup(baseline / avg_total_value).replace("x", r"$\times$")
                best_total = best_by_size.get(size)
                if best_total is not None and abs(avg_total_value - best_total) <= eps:
                    avg_total = r"\textbf{" + avg_total + "}"
            else:
                avg_total = "--"
                speedup = "--"

            return (
                n_text,
                k_text,
                method_text,
                avg_total,
                speedup,
                format_solved_tex(agg),
            )

        for pair_idx, (left_size, right_size) in enumerate(size_pairs):
            for method_idx, method in enumerate(methods):
                left_cells = render_cells(left_size, method, method_idx)
                right_cells = render_cells(right_size, method, method_idx)
                row_cells = list(left_cells) + list(right_cells)
                lines.append(" & ".join(row_cells) + r" \\")
            if pair_idx < len(size_pairs) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(rf"\label{{tab:{problem.lower()}_result}}")
        lines.append(r"\end{table*}")

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    all_expected: List[RowKey] = []
    run_mappings: Dict[str, Dict[int, RowKey]] = {}
    skipped_mappings: Dict[str, set[int]] = {}

    for farm in args.farms:
        table_path = args.cc_root / farm / "table.dat"
        run_to_key, expected, skipped_run_ids = parse_table_for_farm(farm, table_path)
        run_mappings[farm] = run_to_key
        skipped_mappings[farm] = skipped_run_ids
        all_expected.extend(expected)

    expected_set = set(all_expected)
    aggregates: Dict[RowKey, Aggregate] = {key: Aggregate() for key in expected_set}

    for farm in args.farms:
        output_farm_dir = args.outputs_root / farm
        for key, record, run_name, case_stderr, case_stdout, has_frontier in scan_output_runs(
            farm, output_farm_dir, run_mappings[farm], skipped_mappings.get(farm, set())
        ):
            if has_frontier and record is not None:
                metrics = extract_metrics_for_key(
                    key, record, farm=farm, run_hint=f"{run_name}:{key.problem}-{key.n}-{key.k}-{key.method}"
                )
                if metrics is not None:
                    total_wall_s, enum_wall_s = metrics
                    aggregates[key].add(total_wall_s, enum_wall_s)
                    continue
            elif has_frontier:
                warn(f"Frontier exists but usable stats are unavailable for {farm}/{run_name}")

            failure = classify_failure(
                case_stderr=case_stderr,
                case_stdout=case_stdout,
                is_gpu_case=is_gpu_method(key.method),
            )
            if failure == "memory":
                aggregates[key].memory_out_count += 1
            elif failure == "timeout":
                aggregates[key].timeout_count += 1

    ordered_keys = sort_rows(expected_set)
    write_csv(
        args.csv_out,
        ordered_keys,
        aggregates,
        args.precision,
        args.speedup_baseline_method,
    )
    write_tex(
        args.tex_out,
        ordered_keys,
        aggregates,
        args.precision,
        args.speedup_baseline_method,
    )

    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote TeX: {args.tex_out}")
    print(f"Rows: {len(ordered_keys)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
