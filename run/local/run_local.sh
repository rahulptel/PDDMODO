#!/bin/bash
# Local runner for a given table configuration.
# Usage: ./run_local.sh <farm_name_or_path_to_dat>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <farm_name_or_path_to_dat>"
    echo "Examples:"
    echo "  $0 cpu-coup"
    echo "  $0 run/local/cpu-coup.dat"
    exit 1
fi

ARG="$1"

# Resolve the table path
if [ -f "$ARG" ]; then
    TABLE_PATH="$(realpath "$ARG")"
    FARM_NAME="$(basename "$TABLE_PATH" .dat)"
elif [ -f "$SCRIPT_DIR/$ARG.dat" ]; then
    TABLE_PATH="$SCRIPT_DIR/$ARG.dat"
    FARM_NAME="$ARG"
else
    echo "Error: Cannot find table configuration for '$ARG'."
    echo "Expected a file path or a farm name having a corresponding .dat file in $SCRIPT_DIR."
    exit 1
fi

OUTPUTS_DIR="$PROJECT_ROOT/outputs/local/$FARM_NAME"

echo "Running local execution for: $FARM_NAME"
echo "Reading cases from $TABLE_PATH..."

CASE_NUM=0
while IFS= read -r cmd || [ -n "$cmd" ]; do
    # Skip empty lines
    if [ -z "$cmd" ]; then
        continue
    fi
    CASE_NUM=$((CASE_NUM + 1))
    
    RUN_DIR="$OUTPUTS_DIR/RUN$CASE_NUM"
    mkdir -p "$RUN_DIR"
    
    echo "Running Case $CASE_NUM..."
    
    (
        cd "$RUN_DIR"
        if command -v timeout >/dev/null 2>&1; then
            timeout 1h bash -c "$cmd" > case.stdout 2> case.stderr
        else
            bash -c "$cmd" > case.stdout 2> case.stderr
        fi
    )
done < "$TABLE_PATH"

echo "Completed all cases for $FARM_NAME. Outputs saved in $OUTPUTS_DIR"
