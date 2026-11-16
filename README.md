# GainReason

## Quick Start

Run from the repository root using Python 3.10:

```bash
pip install -r requirements.txt
```

- Place Qwen3-1.7B and its tokenizer in `models/Qwen3-1.7B/`. Industrial data is provided in `datasets/`.
- Configure paths, GPU IDs, and batch sizes in both scripts (eight GPUs by default). Set `REPORT_TO="none"` to disable W&B.

```bash
# Stage 1: Itemic Alignment
bash script/itemic.sh
```

Stage 2 loads the Stage 1 model from `checkpoints/<CATEGORY>/itemic/final_checkpoint/`. Provide the offline reasoning JSONL at `REASONING_FILE` (default: `datasets/rec_reasoning_verl/<CATEGORY>/train_reasoning_4step.jsonl`). Required fields: `history_item_sid`, `item_sid`, and `reasoning`.

```bash
# Stage 2: Reasoning Activation
bash script/reasoning_activation.sh
```

Each stage exports its model and tokenizer to `<OUTPUT_DIR>/final_checkpoint/`.

The remaining code is being organized and will be released as soon as possible.
