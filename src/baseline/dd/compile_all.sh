#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

repo_root="$(CDPATH= cd -- "../../.." && pwd)"
bin_dir="$repo_root/resources/bin/baseline/dd"
first_num_objs=3
last_num_objs=7
make_cmd="${MAKE:-make}"
jobs="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 1)}"
make_args=()
num_range_args=0

while (($#)); do
    case "$1" in
        *=*)
            make_args+=("$1")
            ;;
        *)
            if ((num_range_args == 0)); then
                first_num_objs="$1"
            elif ((num_range_args == 1)); then
                last_num_objs="$1"
            else
                echo "Usage: $0 [first-num-objectives] [last-num-objectives] [MAKE_VAR=value ...]" >&2
                exit 2
            fi
            ((++num_range_args))
            ;;
    esac
    shift
done

if ! [[ "$first_num_objs" =~ ^[0-9]+$ && "$last_num_objs" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [first-num-objectives] [last-num-objectives] [MAKE_VAR=value ...]" >&2
    exit 2
fi

if (( first_num_objs > last_num_objs )); then
    echo "first-num-objectives must be less than or equal to last-num-objectives" >&2
    exit 2
fi

for ((num_objs = first_num_objs; num_objs <= last_num_objs; num_objs++)); do
    output="multiobj${num_objs}"

    echo "Building ${output} with NUM_OBJS=${num_objs}"
    mkdir -p "$bin_dir"
    rm -f "$bin_dir/$output"
    "$make_cmd" "${make_args[@]}" clean
    "$make_cmd" -j "$jobs" NUM_OBJS="$num_objs" "${make_args[@]}"
    cp "$bin_dir/multiobj" "$bin_dir/$output"
done
