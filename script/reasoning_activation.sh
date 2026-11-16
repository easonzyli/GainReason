#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# The category and model size must match script/itemic.sh.
CATEGORY="Industrial_and_Scientific"
CUDA_LIST="0,1,2,3,4,5,6,7"
DATA_ROOT="datasets"
SID_INDEX_PATH="${DATA_ROOT}/index/${CATEGORY}.index.json"
ITEM_META_PATH="${DATA_ROOT}/index/${CATEGORY}.item.json"
BASE_MODEL="checkpoints/${CATEGORY}/itemic/final_checkpoint"
OUTPUT_DIR="checkpoints/${CATEGORY}/reasoning_activation"
REASONING_FILE="${DATA_ROOT}/rec_reasoning_verl/${CATEGORY}/train_reasoning_4step.jsonl"
BATCH_SIZE=128
MICRO_BATCH_SIZE=16
NUM_EPOCHS=5
LEARNING_RATE=2e-5
CUTOFF_LEN=1024
SAMPLE=-1
SEED=42
REPORT_TO="none"
RESUME_FROM_CHECKPOINT=""

IFS=',' read -r -a GPUS <<< "${CUDA_LIST}"
NPROC=${#GPUS[@]}
if (( BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC) != 0 )); then
    echo "BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * GPU count" >&2
    exit 1
fi
for path in "${BASE_MODEL}" "${SID_INDEX_PATH}" "${ITEM_META_PATH}"; do
    [[ -e "${path}" ]] || { echo "Missing input: ${path}; run script/itemic.sh first" >&2; exit 1; }
done
[[ -f "${REASONING_FILE}" ]] || { echo "Missing offline reasoning JSONL: ${REASONING_FILE}; upload the prepared file before training." >&2; exit 1; }
resume_options=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_options+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
CUDA_VISIBLE_DEVICES="${CUDA_LIST}" torchrun --standalone --nproc_per_node="${NPROC}" \
    model_preparation/itemic_align.py \
    --training_stage reasoning_activation --warmup_steps 10 \
    --base_model "${BASE_MODEL}" --output_dir "${OUTPUT_DIR}" --category "${CATEGORY}" \
    --sid_index_path "${SID_INDEX_PATH}" \
    --item_meta_path "${ITEM_META_PATH}" \
    --reasoning_train_file "${REASONING_FILE}" \
    --batch_size "${BATCH_SIZE}" --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" --learning_rate "${LEARNING_RATE}" \
    --cutoff_len "${CUTOFF_LEN}" --sample "${SAMPLE}" --seed "${SEED}" \
    --train_from_scratch False --train_new_token_embeddings_only False \
    --report_to "${REPORT_TO}" --wandb_project project_name --wandb_run_name "${CATEGORY}_reasoning_activation" \
    ${resume_options[@]+"${resume_options[@]}"}
