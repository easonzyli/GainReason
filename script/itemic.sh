#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Stage 1: itemic alignment.
# All configured paths are relative to the repository root.
CATEGORY="Industrial_and_Scientific"
CUDA_LIST="0,1,2,3,4,5,6,7"
BASE_MODEL="models/Qwen3-1.7B"
DATA_ROOT="datasets"
TRAIN_FILE="${DATA_ROOT}/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="${DATA_ROOT}/valid/${CATEGORY}_5_2016-10-2018-11.csv"
SID_INDEX_PATH="${DATA_ROOT}/index/${CATEGORY}.index.json"
ITEM_META_PATH="${DATA_ROOT}/index/${CATEGORY}.item.json"
OUTPUT_DIR="checkpoints/${CATEGORY}/itemic"
TRAIN_TASKS="sid_sft,sid_item_feat,fusion_seq_rec,sft_data,title_history2sid"
BATCH_SIZE=128
MICRO_BATCH_SIZE=8
NUM_EPOCHS=10
LEARNING_RATE=1e-4
CUTOFF_LEN=1024
SAMPLE=-1
EVAL_SAMPLE=5000
SEED=42
REPORT_TO="wandb"  # Set to none to disable W&B reporting.
WANDB_PROJECT="project_name"
RUN_NAME="GainReason_${CATEGORY}_itemic_Qwen3-1.7B"
TRAIN_FROM_SCRATCH=False
USE_LORA=False
MASK_ASSISTANT=True
PREPROCESS_NUM_WORKERS=8
RESUME_FROM_CHECKPOINT=""

IFS=',' read -r -a GPUS <<< "${CUDA_LIST}"
NPROC=${#GPUS[@]}
if (( BATCH_SIZE <= 0 || MICRO_BATCH_SIZE <= 0 )); then
    echo "Batch sizes must be positive" >&2
    exit 1
fi
if (( BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC) != 0 )); then
    echo "BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * GPU count" >&2
    exit 1
fi
for path in "${BASE_MODEL}" "${TRAIN_FILE}" "${EVAL_FILE}" "${SID_INDEX_PATH}" "${ITEM_META_PATH}"; do
    [[ -e "${path}" ]] || { echo "Missing input: ${path}; see README.md" >&2; exit 1; }
done
resume_options=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    resume_options+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# Launch the Python training entry directly, with one process per selected GPU.
CUDA_VISIBLE_DEVICES="${CUDA_LIST}" torchrun --standalone --nproc_per_node="${NPROC}" \
    model_preparation/itemic_align.py \
    --training_stage itemic --warmup_steps 20 \
    --base_model "${BASE_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --cutoff_len "${CUTOFF_LEN}" --sample "${SAMPLE}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --generation_eval_sample "${EVAL_SAMPLE}" \
    --output_dir "${OUTPUT_DIR}" \
    --report_to "${REPORT_TO}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}" \
    --category "${CATEGORY}" --seed "${SEED}" \
    --sid_index_path "${SID_INDEX_PATH}" \
    --item_meta_path "${ITEM_META_PATH}" \
    --train_tasks "${TRAIN_TASKS}" \
    --train_from_scratch "${TRAIN_FROM_SCRATCH}" \
    --use_lora "${USE_LORA}" \
    --mask_assistant "${MASK_ASSISTANT}" \
    --preprocess_num_workers "${PREPROCESS_NUM_WORKERS}" \
    ${resume_options[@]+"${resume_options[@]}"}
