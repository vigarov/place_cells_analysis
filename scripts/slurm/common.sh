#!/usr/bin/env bash
# Shared setup for scripts/slurm/*.sh — source after #SBATCH lines, not executed directly.
set -euo pipefail

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SLURM_DIR}/../.." && pwd)"

# Load paths from .env. Override before sourcing, or set ENV_FILE in the environment.
ENV_FILE="${ENV_FILE:-${_REPO_ROOT}/.env}"
if [[ ! -f "${ENV_FILE}" ]]; then
	echo "ERROR: env file not found: ${ENV_FILE}" >&2
	echo "Copy .env.example to .env and fill in the placeholders." >&2
	exit 1
fi
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${PROJECT_ROOT:?Set PROJECT_ROOT in ${ENV_FILE}}"
: "${SLURM_LOG_DIR:?Set SLURM_LOG_DIR in ${ENV_FILE}}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in ${ENV_FILE}}"

PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
mkdir -p "${SLURM_LOG_DIR}"

cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

_UV_ENV_READY=false

_run_uv_setup() {
	module purge
	module load CUDA/12.6.0 || true
	export _REPO_ROOT
	"${_REPO_ROOT}/setup.sh"
}

run_uv() {
	if [[ "${_UV_ENV_READY}" != true ]]; then
		_run_uv_setup
		_UV_ENV_READY=true
	fi
	uv run "$@"
}

_slurm_stage_from_script() {
	basename "${1:?}" .sh
}

# Set SLURM_OUTPUT_PATH / SLURM_ERROR_PATH for sbatch --output / --error.
slurm_log_paths() {
	local stage="${1:?}"
	local mode="${2:-single}"

	if [[ "${mode}" == "array" ]]; then
		SLURM_OUTPUT_PATH="${SLURM_LOG_DIR}/${stage}_%A_%a.out"
		SLURM_ERROR_PATH="${SLURM_LOG_DIR}/${stage}_%A_%a.err"
	else
		SLURM_OUTPUT_PATH="${SLURM_LOG_DIR}/${stage}_%j.out"
		SLURM_ERROR_PATH="${SLURM_LOG_DIR}/${stage}_%j.err"
	fi
}

slurm_log_cli_args() {
	local stage="${1:?}"
	local mode="${2:-single}"
	slurm_log_paths "${stage}" "${mode}"
	printf '%s\n' "--output=${SLURM_OUTPUT_PATH}" "--error=${SLURM_ERROR_PATH}"
}

_sbatch_log_and_array_args() {
	local script_path="${1:?}"
	local stage="${2:-$(_slurm_stage_from_script "${script_path}")}"
	local -a args=()

	mapfile -t args < <(slurm_log_cli_args "${stage}" single)
	printf '%s\0' "${args[@]}"
}

parse_submit_args() {
	while (($#)); do
		case "$1" in
		--no-auto-sync | --auto-sync)
			shift
			if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
				shift
			fi
			;;
		*)
			echo "ERROR: unknown submit argument: $1" >&2
			echo "Usage: bash $0" >&2
			exit 2
			;;
		esac
	done
}

_wait_for_slurm_job() {
	local job_id="${1:?}"
	echo "Waiting for Slurm job ${job_id}..."
	while squeue -h -j "${job_id}" 2>/dev/null | grep -q .; do
		sleep 5
	done

	local failed
	failed="$(sacct -j "${job_id}" -X --format=State -n 2>/dev/null | grep -Ev 'COMPLETED|COMPLETING' | grep -Ev '^$' || true)"
	if [[ -n "${failed}" ]]; then
		echo "ERROR: job ${job_id} did not complete successfully:" >&2
		sacct -j "${job_id}" -X --format=JobID,State,ExitCode -n 2>/dev/null >&2 || true
		return 1
	fi
	echo "Job ${job_id} completed successfully."
	return 0
}

_sbatch_with_logs() {
	local script_path="${1:?}"
	shift
	local stage="${1:-}"
	local -a script_args=()
	if [[ -n "${stage}" && "${stage}" != --* ]]; then
		shift
	else
		stage="$(_slurm_stage_from_script "${script_path}")"
	fi

	local -a sbatch_args=(--parsable --account="${SLURM_ACCOUNT}" --chdir="${PROJECT_ROOT}")
	mapfile -d '' -t _log_array_args < <(_sbatch_log_and_array_args "${script_path}" "${stage}")
	sbatch_args+=("${_log_array_args[@]}")
	if (($#)); then
		if [[ "$1" == "--" ]]; then
			shift
			script_args=("$@")
		else
			sbatch_args+=("$@")
		fi
	fi
	if ((${#script_args[@]})); then
		sbatch "${sbatch_args[@]}" "${script_path}" -- "${script_args[@]}"
	else
		sbatch "${sbatch_args[@]}" "${script_path}"
	fi
}

_run_slurm_stage() {
	local script_path="${1:?}"
	shift
	local stage="$(_slurm_stage_from_script "${script_path}")"
	local job_id rc=0

	job_id="$(_sbatch_with_logs "${script_path}" "${stage}" "$@")"
	LAST_SLURM_JOB_ID="${job_id}"
	echo "Submitted ${job_id} (${script_path}, logs under ${SLURM_LOG_DIR})"

	_wait_for_slurm_job "${job_id}" || rc=$?
	return "${rc}"
}

_run_submit_wrapper() {
	local script_path="${1:?}"
	shift
	parse_submit_args "$@"
	_run_slurm_stage "${script_path}"
	exit $?
}
