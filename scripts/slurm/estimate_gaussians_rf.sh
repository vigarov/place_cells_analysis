#!/usr/bin/env bash
# Submit one Slurm job per pending optimizer Gaussian-RF fit.
#
# From login node:
#   bash scripts/slurm/estimate_gaussians_rf.sh INPUT [estimate-gaussians-rf args...]
#
# INPUT: experiment suffix directory, e.g.
#   results/single_room/600s_warm15x60s_train10x10s_ep1


#SBATCH --job-name=pc-gauss-rf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=himem


resolve_gaussians_input() {
	local input="${1:-}"
	if [[ -z "${input}" ]]; then
		echo "ERROR: INPUT path required." >&2
		echo "Usage: bash $0 INPUT [estimate-gaussians-rf args...]" >&2
		exit 2
	fi
	if [[ "${input}" != /* ]]; then
		input="${PROJECT_ROOT}/${input}"
	fi
	if [[ ! -d "${input}" ]]; then
		echo "ERROR: input directory not found: ${input}" >&2
		exit 1
	fi
	GAUSSIANS_INPUT="${input}"
}

require_optimizer_arg() {
	local has_optimizer=false
	local arg
	for arg in "$@"; do
		if [[ "${arg}" == "--optimizer" ]]; then
			has_optimizer=true
			break
		fi
	done
	if [[ "${has_optimizer}" != true ]]; then
		echo "ERROR: worker job requires --optimizer TAG" >&2
		exit 2
	fi
}

if [[ -z "${SLURM_JOB_ID:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
	set -euo pipefail
	# shellcheck source=common.sh
	source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

	GAUSSIANS_INPUT=""
	resolve_gaussians_input "${1:-}"
	shift || true
	EXTRA_ARGS=("$@")

	mapfile -t PENDING < <(
		uv run estimate-gaussians-rf \
			--input "${GAUSSIANS_INPUT}" \
			--list-pending \
			"${EXTRA_ARGS[@]}"
	)

	if ((${#PENDING[@]} == 0)); then
		echo "No pending optimizer fits under ${GAUSSIANS_INPUT}"
		exit 0
	fi

	echo "Scheduling ${#PENDING[@]} optimizer job(s) under ${GAUSSIANS_INPUT}:"
	printf '  %s\n' "${PENDING[@]}"

	for opt in "${PENDING[@]}"; do
		stage="estimate_gaussians_rf_${opt//./_}"
		job_id="$(_sbatch_with_logs "${BASH_SOURCE[0]}" "${stage}" \
			"${GAUSSIANS_INPUT}" \
			--optimizer "${opt}" \
			--n-processes 60 \
			"${EXTRA_ARGS[@]}")"
		echo "Submitted ${job_id} (--optimizer ${opt})"
	done
	exit 0
fi

set -euo pipefail
source scripts/slurm/common.sh

GAUSSIANS_INPUT=""
resolve_gaussians_input "${1:-}"
shift || true
require_optimizer_arg "$@"

echo "Gaussian RF fits input=${GAUSSIANS_INPUT} $*"

run_uv estimate-gaussians-rf --input "${GAUSSIANS_INPUT}" "$@"
