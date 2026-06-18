#!/usr/bin/env bash
# From login node: bash scripts/slurm/cycles_experiment.sh CONFIG.json
#
# CONFIG is required (relative to PROJECT_ROOT or absolute path).

#SBATCH --job-name=pc-cycles
#SBATCH --time=32:00:00 # will depend on the step_size and GPU compute power, in particular, we have 1~= 230s; 20 ~= 210s/room ; 316 ~= 69s/room ; 908 ~= 58s/room ; 1204 ~= 91s/room ; 1500 ~= 129s/room on an L4 GPU
#SBATCH --gpus=1
#SBATCH --partition=gpu_rtx8000_48gb,gpu_v100_32gb,gpu_a100_40gb,gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

resolve_cycles_config() {
	local config="${1:-}"
	if [[ -z "${config}" ]]; then
		echo "ERROR: config path required." >&2
		echo "Usage: bash $0 CONFIG.json" >&2
		exit 2
	fi
	if [[ "${config}" != /* ]]; then
		config="${PROJECT_ROOT}/${config}"
	fi
	if [[ ! -f "${config}" ]]; then
		echo "ERROR: cycles config not found: ${config}" >&2
		exit 1
	fi
	CYCLES_CONFIG="${config}"
}

if [[ -z "${SLURM_JOB_ID:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
	set -euo pipefail
	# shellcheck source=common.sh
	source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
	resolve_cycles_config "${1:-}"
	_run_slurm_stage "${BASH_SOURCE[0]}" "${CYCLES_CONFIG}"
	exit $?
fi

set -euo pipefail
source scripts/slurm/common.sh
resolve_cycles_config "${1:-}"

echo "cycles experiment config=${CYCLES_CONFIG}"

run_uv cycles-experiment --config "${CYCLES_CONFIG}"
