#!/usr/bin/env bash
set -euo pipefail

gpu_count="${1:-1}"
seed_count="${2:-10}"
python_bin="${PYTHON_BIN:-python}"
config_file="${CONFIG_FILE:-configs/GPS/direct_train.yaml}"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root="${root_dir}/.parallel_dataset_cache"
experiment_name="${EXPERIMENT_NAME:-fifth_component_weight2}"
log_dir="${root_dir}/logs/${experiment_name}"

if (( gpu_count < 1 )); then
    echo "gpu_count must be at least 1" >&2
    exit 1
fi

mkdir -p "${cache_root}" "${log_dir}"
cd "${root_dir}"

for ((seed = 0; seed < seed_count; seed++)); do
    while (( $(jobs -rp | wc -l) >= gpu_count )); do
        wait -n
    done

    gpu_index=$((seed % gpu_count))
    cache_dir="${cache_root}/seed_${seed}"
    mkdir -p "${cache_dir}"
    if [[ ! -e "${cache_dir}/raw" ]]; then
        ln -s "${root_dir}/datasets_lrx/raw" "${cache_dir}/raw"
    fi

    CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" main.py \
        --cfg "${config_file}" --repeat 1 \
        seed "${seed}" dataset.dir "${cache_dir}" \
        > "${log_dir}/seed_${seed}.log" 2>&1 &
done

wait
