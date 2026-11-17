# GainReason

This repository contains the official implementation of **“Learning to Reason with Predictive Gains for Generative Recommendation”**.

## 🚀 Quick Start

Run all commands from the repository root.

### 🛠️ Environment

```bash
conda create -n gainreason python=3.10 -y
conda activate gainreason
python -m pip install -r requirements.txt gdown
```

### 📦 Data and Models

Download `Amazon_data.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/1mkV_cr9Df90yogoX2bEo9bRYraB3exDM?usp=drive_link) and **replace the existing data in `datasets/`** with its contents:

```bash
mkdir -p downloads datasets
gdown --folder "https://drive.google.com/drive/folders/1mkV_cr9Df90yogoX2bEo9bRYraB3exDM?usp=drive_link" -O downloads/
tar -xzf downloads/Amazon_data.tar.gz -C datasets/
```

- Place Qwen3-1.7B and its tokenizer in `models/Qwen3-1.7B/`.
- Set `CATEGORY` to `Industrial_and_Scientific`, `Movies`, or `Yelp` in both scripts. Configure paths, GPU IDs, and batch sizes (eight GPUs by default). Set `REPORT_TO="none"` to disable W&B.

## 🧠 Model Preparation

**1. Itemic Alignment**

```bash
bash script/itemic.sh
```

The best-validation-loss model is exported to `checkpoints/<CATEGORY>/itemic/final_checkpoint/`.

**2. Reasoning Activation**

Continue training from the Stage 1 checkpoint:

```bash
bash script/reasoning_activation.sh
```

**Note:** Provide the offline reasoning JSONL at `REASONING_FILE` (default: `datasets/rec_reasoning_verl/<CATEGORY>/train_reasoning_4step.jsonl`). Required fields: `history_item_sid`, `item_sid`, and `reasoning`.

Each stage exports its model and tokenizer to `<OUTPUT_DIR>/final_checkpoint/`.

## 🚧 Code Release

The remaining code is being organized and will be released as soon as possible.
