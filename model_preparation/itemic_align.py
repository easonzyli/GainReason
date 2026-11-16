import os

import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from peft import LoraConfig, TaskType, TrainableTokensConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
import re
from collections import Counter
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import wandb
from contextlib import contextmanager
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_Qwen3 import (
    SFTData,
    SidSFTDataset,
    SidItemFeatDataset,
    FusionSeqRecDataset,
    TitleHistory2SidSFTDataset,
    EvalSidDataset,
    ReasoningActivationDataset,
)
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset

import os


SID_PATTERN = re.compile(r"<a_\d+><b_\d+><c_\d+>")


def _normalize_text_tokens(text):
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _token_f1(prediction, target):
    pred_tokens = _normalize_text_tokens(prediction)
    target_tokens = _normalize_text_tokens(target)
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    target_counter = Counter(target_tokens)
    overlap = sum((pred_counter & target_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_mapping_outputs(text):
    outputs = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if "->" in line:
            outputs.append(line.split("->", 1)[1].strip())
        elif ":" in line:
            outputs.append(line.split(":", 1)[1].strip())
        else:
            outputs.append(line)
    return outputs


def _compute_sid_meaning_token_f1(predictions, records):
    scores = []
    for prediction, record in zip(predictions, records):
        pred_titles = _extract_mapping_outputs(prediction)
        target_titles = record["target_titles"]
        for idx, target_title in enumerate(target_titles):
            pred_title = pred_titles[idx] if idx < len(pred_titles) else ""
            scores.append(_token_f1(pred_title, target_title))
    return float(np.mean(scores)) if scores else 0.0


def _compute_title_to_sid_accuracy(predictions, records):
    correct = 0
    total = 0
    for prediction, record in zip(predictions, records):
        pred_sids = SID_PATTERN.findall(str(prediction))
        target_sids = record["target_sids"]
        for idx, target_sid in enumerate(target_sids):
            pred_sid = pred_sids[idx] if idx < len(pred_sids) else ""
            correct += int(pred_sid == target_sid)
            total += 1
    return correct / total if total else 0.0


class MultiEvalTrainer(transformers.Trainer):
    """
    Runs the default evaluation and then iterates through any extra eval sets so every epoch
    produces loss numbers for the auxiliary datasets as well.
    """

    def __init__(
        self,
        *args,
        extra_eval_sets: Optional[Dict[str, HFDataset]] = None,
        generation_eval_sets: Optional[Dict[str, Any]] = None,
        generation_tokenizer=None,
        generation_max_new_tokens: int = 256,
        **kwargs,
    ):
        self.extra_eval_sets = extra_eval_sets or {}
        self.generation_eval_sets = generation_eval_sets or {}
        self.generation_tokenizer = generation_tokenizer
        self.generation_max_new_tokens = generation_max_new_tokens
        super().__init__(*args, **kwargs)

    def _save_rng_state(self, output_dir):
        # On OSS/FUSE mounts, concurrent os.makedirs(exist_ok=True) from all ranks
        # can still raise FileExistsError because the post-mkdir isdir() check does
        # not see the freshly created dir (eventual consistency). Serialize dir
        # creation on the main process, sync with a barrier, and retry on EEXIST.
        import time as _time

        if self.args.process_index == 0:
            os.makedirs(output_dir, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        for attempt in range(6):
            try:
                super()._save_rng_state(output_dir)
                return
            except FileExistsError:
                if attempt == 5:
                    raise
                _time.sleep(2 * (attempt + 1))

    @contextmanager
    def _disable_callback(self, callback_cls):
        callbacks = self.callback_handler.callbacks
        removed = [cb for cb in callbacks if isinstance(cb, callback_cls)]
        if not removed:
            yield
            return
        self.callback_handler.callbacks = [cb for cb in callbacks if not isinstance(cb, callback_cls)]
        try:
            yield
        finally:
            self.callback_handler.callbacks = callbacks

    def evaluate(
        self,
        eval_dataset: Optional[HFDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        for name, dataset in self.extra_eval_sets.items():
            if dataset is None:
                continue
            with self._disable_callback(EarlyStoppingCallback):
                extra_metrics = super().evaluate(
                    eval_dataset=dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{name}",
                )
            self.log(extra_metrics)
            metrics.update(extra_metrics)
        if self.generation_eval_sets:
            generation_metrics = self._evaluate_generation_sets(metric_key_prefix)
            self.log(generation_metrics)
            metrics.update(generation_metrics)
        return metrics

    def _generate_for_records(self, records):
        if self.generation_tokenizer is None:
            return []
        model = self.model
        model.eval()
        predictions = []
        device = model.device
        pad_token_id = self.generation_tokenizer.pad_token_id
        eos_token_id = self.generation_tokenizer.eos_token_id
        for record in records:
            input_ids = torch.tensor([record["prompt_input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            generated_ids = output_ids[0, input_ids.shape[-1]:]
            prediction = self.generation_tokenizer.decode(generated_ids, skip_special_tokens=True)
            predictions.append(prediction.strip())
        return predictions

    def _evaluate_generation_sets(self, metric_key_prefix):
        metrics = {}
        for name, dataset in self.generation_eval_sets.items():
            records = dataset.get_eval_records()
            predictions = self._generate_for_records(records)
            if name == "sid_sequence_meaning":
                metrics[f"{metric_key_prefix}_{name}_token_f1"] = _compute_sid_meaning_token_f1(predictions, records)
            elif name == "title_sequence_sid":
                metrics[f"{metric_key_prefix}_{name}_item_accuracy"] = _compute_title_to_sid_accuracy(predictions, records)
        return metrics

    def training_step(self, model, inputs, *args, **kwargs):
        task_ids = inputs.pop("task_id", None)
        task_names = inputs.pop("task_name", None)
        loss = super().training_step(model, inputs, *args, **kwargs)
        return loss


class TaskAwareSeq2SeqCollator:
    def __init__(self, tokenizer, **kwargs):
        self.base_collator = transformers.DataCollatorForSeq2Seq(tokenizer, **kwargs)

    def __call__(self, features):
        has_task_info = any("task_id" in feature or "task_name" in feature for feature in features)
        task_ids = [feature.pop("task_id", -1) for feature in features]
        task_names = [feature.pop("task_name", "unknown") for feature in features]
        batch = self.base_collator(features)
        if has_task_info:
            batch["task_id"] = torch.tensor(task_ids, dtype=torch.long)
            batch["task_name"] = task_names
        return batch


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _decode_tokens(tokens, tokenizer_ref):
    if not isinstance(tokens, (list, tuple)):
        return ""
    valid_ids = [tid for tid in tokens if isinstance(tid, int) and tid >= 0]
    if not valid_ids:
        return ""
    return tokenizer_ref.decode(valid_ids, skip_special_tokens=False)

def _preview_dataset(dataset, name, tokenizer_ref, max_samples=3):
    print(f"[Preview] {name}: displaying up to {max_samples} samples")
    preview_count = min(max_samples, len(dataset))
    for idx in range(preview_count):
        sample = dataset[idx]
        input_text = ""
        # label_text = ""
        if isinstance(sample, dict):
            if "input_ids" in sample:
                input_text = _decode_tokens(sample["input_ids"], tokenizer_ref)
            if "labels" in sample:
                # Filter label padding tokens (e.g., -100) before decoding for readability
                label_ids = [tid for tid in sample["labels"] if isinstance(tid, int) and tid >= 0]
                label_text = _decode_tokens(label_ids, tokenizer_ref)
        print(f"Sample {idx + 1}:")
        if input_text:
            print(f"  Input : {input_text}")
        if label_text:
            print(f"  Label : {label_text}")
            print(f"  Length: {len(label_ids)} tokens")
        print()



def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


DEFAULT_TRAIN_TASKS = [
    "sid_sft",
    "sid_item_feat",
    "fusion_seq_rec",
    "sft_data",
    "title_history2sid",
]


TASK_ALIASES = {
    "sidsftdataset": "sid_sft",
    "sid_sft": "sid_sft",
    "sid": "sid_sft",
    "siditemfeatdataset": "sid_item_feat",
    "sid_item_feat": "sid_item_feat",
    "item_feat": "sid_item_feat",
    "sid2title": "sid_item_feat",
    "title2sid": "sid_item_feat",
    "fusionseqrecdataset": "fusion_seq_rec",
    "fusion_seq_rec": "fusion_seq_rec",
    "fusion": "fusion_seq_rec",
    "sftdata": "sft_data",
    "sft_data": "sft_data",
    "text_rec": "sft_data",
    "titlehistory2sidsftdataset": "title_history2sid",
    "title_history2sid": "title_history2sid",
    "title2sid_rec": "title_history2sid",
}


def _normalize_train_tasks(train_tasks):
    if train_tasks is None or train_tasks == "" or train_tasks == "all":
        return list(DEFAULT_TRAIN_TASKS)

    if isinstance(train_tasks, str):
        raw_tasks = train_tasks.strip()
        if raw_tasks.lower() == "all":
            return list(DEFAULT_TRAIN_TASKS)
        if raw_tasks.startswith("["):
            raw_tasks = json.loads(raw_tasks)
        else:
            raw_tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]

    elif isinstance(train_tasks, (list, tuple, set)):
        raw_tasks = list(train_tasks)
    else:
        raise ValueError(f"Unsupported train_tasks type: {type(train_tasks)}")

    normalized_tasks = []
    unknown_tasks = []
    for task in raw_tasks:
        task_key = str(task).strip().lower().replace("-", "_")
        task_key = TASK_ALIASES.get(task_key, task_key)
        if task_key not in DEFAULT_TRAIN_TASKS:
            unknown_tasks.append(str(task))
            continue
        if task_key not in normalized_tasks:
            normalized_tasks.append(task_key)

    if unknown_tasks:
        raise ValueError(
            f"Unknown train_tasks: {unknown_tasks}. Supported tasks: {DEFAULT_TRAIN_TASKS}"
        )
    if not normalized_tasks:
        raise ValueError("train_tasks is empty after normalization.")
    return normalized_tasks


def train(
    training_stage: str = "itemic",
    reasoning_train_file: str = "",
    warmup_steps: Optional[int] = None,
    # model/data params
    base_model: str = "models/Qwen3-1.7B",
    train_file: str = "datasets/train/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    eval_file: str = "datasets/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv",
    output_dir: str = "checkpoints/Industrial_and_Scientific/itemic",
    sample: int = -1,
    seed: int = 42,
    category: str = "Industrial_and_Scientific",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "Industrial_and_Scientific_stage1_sft_Qwen3-1.7B",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    train_from_scratch: bool = False,
    sid_index_path: str = "datasets/index/Industrial_and_Scientific.index.json",
    item_meta_path: str = "datasets/index/Industrial_and_Scientific.item.json",
    llm_generated_data_path: str = "datasets/index/Industrial_and_Scientific.item_enhanced_v2.json",
    llm_generated_sequence_path: str = "datasets/index/Industrial_and_Scientific.integrated_narrative.csv",
    mask_assistant: bool = True,   # Whether only the target response is used for loss calculation
    train_new_token_embeddings_only: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Union[str, List[str]] = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    train_tasks: str = None,
    generation_eval_sample: int = 5000,
    preprocess_num_workers: int = 1,
    report_to: str = "none",
):
    if training_stage not in ("itemic", "reasoning_activation"):
        raise ValueError("training_stage must be itemic or reasoning_activation")
    is_itemic = training_stage == "itemic"
    if not is_itemic and not reasoning_train_file:
        raise ValueError("reasoning_train_file is required for reasoning_activation")
    if not is_itemic and (batch_size <= 0 or micro_batch_size <= 0):
        raise ValueError("batch sizes must be positive")
    if warmup_steps is None:
        warmup_steps = 20 if is_itemic else 10
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if not is_itemic:
        # Disable NCCL peer-to-peer transport during reasoning activation.
        os.environ["NCCL_P2P_DISABLE"] = "1"
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project
    os.environ["SIDREASONER_PREPROCESS_NUM_WORKERS"] = str(preprocess_num_workers)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Yelp": "local businesses", "Movies_and_TV": "movies"}
    print(category)
    if category in category_dict:
        category = category_dict[category]
    else:
        category = "items"

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not is_itemic and batch_size % (micro_batch_size * world_size):
        raise ValueError("batch_size must be divisible by micro_batch_size * WORLD_SIZE")
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Tokenizer length: {len(tokenizer)}")
    
    boundary_tokens = ["<|sid_begin|>", "<|sid_end|>"]
    candidate_tokens = []
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0]
        )
        new_tokens = token_extender.get_new_tokens()
        candidate_tokens.extend(new_tokens)
    else:
        new_tokens = []
    candidate_tokens.extend(boundary_tokens)

    existing_vocab = set(tokenizer.get_vocab().keys())
    tokens_to_add = [tok for tok in candidate_tokens if tok not in existing_vocab]
    if tokens_to_add:
        print(f"Adding {len(tokens_to_add)} new tokens to tokenizer")
        tokenizer.add_tokens(tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))
        num_new_tokens = len(tokens_to_add)
    else:
        print("All candidate tokens already exist in the tokenizer; skipping addition.")
        num_new_tokens = 0

    if use_lora and train_new_token_embeddings_only:
        raise ValueError("use_lora and train_new_token_embeddings_only cannot both be True.")

    if use_lora:
        if isinstance(lora_target_modules, str):
            target_modules = [m.strip() for m in lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = list(lora_target_modules)
        print(
            f"LoRA fine-tuning enabled: r={lora_r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout}, target_modules={target_modules}"
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif train_new_token_embeddings_only:
        if num_new_tokens > 0:
            vocab_size = len(tokenizer)
            new_token_indices = list(range(vocab_size - num_new_tokens, vocab_size))
            print(f"Restricting training to new token ids.")
            peft_config = TrainableTokensConfig(
                token_indices=new_token_indices,
                target_modules=["embed_tokens"],
                init_weights=True,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        print("Full fine-tuning enabled: attention blocks, FFNs, and embeddings remain trainable.")

    if num_new_tokens == 0 and train_new_token_embeddings_only:
        print("No new tokens added; the entire model will remain trainable.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
    print(f"Trainable parameters: {trainable_params} / {total_params} ({percent:.4f}%)")
        
    if is_itemic:
        selected_train_tasks = _normalize_train_tasks(train_tasks)
        selected_train_task_set = set(selected_train_tasks)
        train_datasets = []
        train_dataset_names = []
        print(f"Selected train tasks: {selected_train_tasks}")
        if "sid_sft" in selected_train_task_set:
            train_datasets.append(SidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant))
            train_dataset_names.append("SidSFTDataset")
        if "sid_item_feat" in selected_train_task_set:
            train_datasets.append(SidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant))
            train_dataset_names.append("SidItemFeatDataset")
        if "fusion_seq_rec" in selected_train_task_set:
            train_datasets.append(FusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant))
            train_dataset_names.append("FusionSeqRecDataset")
        if "sft_data" in selected_train_task_set:
            train_datasets.append(SFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant))
            train_dataset_names.append("SFTData")
        if "title_history2sid" in selected_train_task_set:
            train_datasets.append(TitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant))
            train_dataset_names.append("TitleHistory2SidSFTDataset")

        if not train_datasets:
            raise ValueError(f"No training datasets were built from train_tasks={selected_train_tasks}")
    
        train_data = ConcatDataset(train_datasets)
    else:
        train_data = ReasoningActivationDataset(
            reasoning_train_file=reasoning_train_file,
            item_file=item_meta_path, index_file=sid_index_path,
            tokenizer=tokenizer, max_len=cutoff_len, sample=sample,
            seed=seed, category=category,
        )
        train_datasets = [train_data]
        train_dataset_names = ["ReasoningActivationDataset"]

    main_rank = int(os.environ.get("RANK", 0)) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0
    if main_rank:
        for ds, name in zip(train_datasets, train_dataset_names):
            _preview_dataset(ds, name, tokenizer)

    val_data_eval_sid = None
    if is_itemic:
        val_data_eval_sid = EvalSidDataset(
            train_file=eval_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=generation_eval_sample,
            seed=seed,
            category=category,
            test=False,
            mask_assistant=True,
        )

    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    hf_val_dataset = None
    if is_itemic:
        sample_frac = 1
        train_records = []
        for task_id, (dataset, task_name) in enumerate(zip(train_datasets, train_dataset_names)):
            for sample_idx in range(len(dataset)):
                record = dict(dataset[sample_idx])
                if record is None:
                    continue
                record["task_id"] = task_id
                record["task_name"] = task_name
                train_records.append(record)
        if not train_records:
            raise ValueError("No train records were built.")
        hf_train_dataset = HFDataset.from_list(train_records)
        hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(int(sample_frac * len(hf_train_dataset))))
        hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data_eval_sid] for k in val_data_eval_sid[0].keys()}).shuffle(seed=seed)
        hf_val_dataset = hf_val_dataset.shuffle(seed=42)
    else:
        if len(train_data) == 0:
            raise ValueError("Reasoning dataset contains no usable examples")
        hf_train_dataset = HFDataset.from_dict({
            key: [row[key] for row in train_data]
            for key in train_data[0].keys()
        })
        hf_train_dataset = hf_train_dataset.shuffle(seed=42).select(range(len(hf_train_dataset)))
    extra_eval_sets = None
    print(hf_train_dataset)
    if is_itemic:
        print(hf_val_dataset)
        print(f"Main eval EvalSidDataset: {len(val_data_eval_sid)}")
    eval_options = {}
    if is_itemic:
        eval_options = dict(
            per_device_eval_batch_size=max(1, micro_batch_size // 2),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    trainer = MultiEvalTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        extra_eval_sets=extra_eval_sets,
        generation_tokenizer=tokenizer,
        generation_max_new_tokens=128,
        args=transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch" if is_itemic else "no",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=is_itemic,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=[] if report_to == "none" else [report_to],
            remove_unused_columns=not is_itemic,
            **eval_options,
        ),
        data_collator=(
            TaskAwareSeq2SeqCollator(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
            if is_itemic else
            transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if is_itemic else [],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    model.state_dict()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
