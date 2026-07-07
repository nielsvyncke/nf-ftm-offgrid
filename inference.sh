#!/bin/bash

# ===============================
# SLURM Job Configuration Example
# ===============================
#SBATCH --account=<account>
#SBATCH --clusters=<cluster>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=<partition>

set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [--conda_path <conda_path>] [--env <env_name>] [--bundle_dir <bundle_directory>] [additional MONAI bundle arguments]

Runs model inference on test data using the MONAI Bundle framework.

Options:
  --conda_path      Path to conda installation to prepend to PATH [optional]
  --env             Name of conda environment to activate [optional]
  --bundle_dir      Path to the MONAI bundle directory [default: one level up from script location]
  -h, --help        Show this help message

MONAI Bundle Arguments:
  Any additional arguments will be forwarded to 'monai.bundle run'.

Prerequisites:
  Ensure the following prerequisites are met before running this script:
    1. Environment Setup:
       $ bash setup.sh --device cuda --env <env_name> --python_version 3.12
    2. Data Download and datalist generation (see README.md).

Examples:
  # Run inference with specified conda environment and bundle directory:
  $0 --env <env_name> --bundle_dir /path/to/bundle --data_dir /path/to/data

SLURM Configuration Example:
  The script includes an example SLURM configuration at the top. Adjust the
  account, cluster, partition, and resource parameters according to your
  cluster settings.

SLURM Submission Example:
  To submit this script as a SLURM job with customized options:
    sbatch --job-name=inference_glas inference.sh --conda_path /path/to/conda --env <env_name> --data_dir /path/to/data

EOF
    exit 1
}

# =======================
# Argument Parsing
# =======================
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
    --conda_path)
        CONDA_PATH="$2"
        shift 2
        ;;
    --env)
        ENV="$2"
        shift 2
        ;;
    --bundle_dir)
        BUNDLE_DIR="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

# Set default bundle directory (one level up from script directory)
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# =======================
# Conda Environment Setup
# =======================
# Prepend specified conda path to PATH if provided
if [[ -n "${CONDA_PATH:-}" ]]; then
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# Initialize conda and activate the specified environment if provided
if [[ -n "${ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV"
fi

# =======================
# Environment Variables
# =======================
echo "Bundle root directory: $BUNDLE_DIR"
export PYTHONPATH="$BUNDLE_DIR"

# =======================
# Main Execution
# =======================
python -m monai.bundle run \
    --meta_file "$BUNDLE_DIR/configs/metadata.json" \
    --config_file "$BUNDLE_DIR/configs/inference.yaml" \
    --bundle_root "$BUNDLE_DIR" \
    "${ARGS[@]}"
