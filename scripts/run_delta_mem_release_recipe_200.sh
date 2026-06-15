#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"

echo "run_delta_mem_release_recipe_200.sh is kept as a compatibility wrapper."
echo "Launching the original delta-Mem baseline: q/k/v/o projections, CE only, no KL."
exec bash "${ROOT_DIR}/scripts/run_delta_mem_original_baseline_200.sh"
