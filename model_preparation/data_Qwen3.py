import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from prompts import (
    SFTData_prompts,
    SidSFTDataset_prompts,
    SidItemFeatDataset_prompts,
    FusionSeqRecDataset_prompts,
    TitleHistory2SidSFTDataset_prompts,
    all_prompt,
)

# ASSISTANT_PREFIX = "<|im_start|>assistant"
ASSISTANT_PREFIX = "</think>"
PROMPT_SYSTEM = "You are a helpful assistant.\n"


def _select_prompt(prompt_pool, idx):
    if isinstance(prompt_pool, dict):
        keys = list(prompt_pool.keys())
        if not keys:
            raise ValueError("empty prompt pool")
        return prompt_pool[random.choice(keys)]
    if not prompt_pool:
        raise ValueError("empty prompt pool")
    return random.choice(prompt_pool)


def _format_prompt(prompt_template, **kwargs):
    return {
        "instruction": prompt_template["instruction"].format(**kwargs),
        "response": prompt_template["response"].format(**kwargs),
    }


def _build_messages(prompt, assistant_response="", include_assistant=True):
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": str(prompt)},
    ]
    if include_assistant:
        messages.append(
            {"role": "assistant", "content": str(assistant_response)}
        )
    return messages


def _get_preprocess_num_workers():
    try:
        return max(1, int(os.environ.get("SIDREASONER_PREPROCESS_NUM_WORKERS", "1")))
    except ValueError:
        return 1


def _disable_preprocess_tqdm():
    return os.environ.get("SIDREASONER_DISABLE_PREPROCESS_TQDM", "").lower() in {"1", "true", "yes", "on"}


def _log_preprocess(message):
    if _disable_preprocess_tqdm():
        print(message, flush=True)


def _is_main_process():
    return int(os.environ.get("RANK", 0)) == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0


def _decode_ids(tokenizer, ids):
    if tokenizer is None:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


def _print_dataset_sample(dataset_name, tokenizer, sample):
    if not _is_main_process() or tokenizer is None or not sample:
        return
    try:
        input_ids = sample.get("input_ids", [])
        input_text = _decode_ids(tokenizer, input_ids)
        labels = sample.get("labels")
        prompt_context_text = ""
        label_text = ""
        if labels is not None:
            prompt_context_ids = [
                input_id for input_id, label_id in zip(input_ids, labels) if label_id == -100
            ]
            prompt_context_text = _decode_ids(tokenizer, prompt_context_ids)
            label_ids = [token_id for token_id in labels if token_id != -100]
            label_text = _decode_ids(tokenizer, label_ids)
        print(f"\n========== {dataset_name} sample ==========", flush=True)
        if prompt_context_text:
            print("[decoded prompt context, labels masked out]", flush=True)
            print(prompt_context_text, flush=True)
        print("[decoded full training sequence]", flush=True)
        print(input_text, flush=True)
        if label_text:
            print("[decoded labels]", flush=True)
            print(label_text, flush=True)
        print(f"========== end {dataset_name} sample ==========\n", flush=True)
    except Exception as exc:
        print(f"[Data sample] Failed to print {dataset_name} sample: {exc}", flush=True)


def _sample_dataframe(data, sample, seed):
    if sample <= 0:
        return data
    return data.sample(min(sample, len(data)), random_state=seed)


def _sample_list(data, sample):
    if sample <= 0 or sample >= len(data):
        return data
    return random.sample(data, sample)


def _collect_preprocessed_inputs(preprocess_fn, total, keep_none=False):
    workers = _get_preprocess_num_workers()
    dataset_name = getattr(getattr(preprocess_fn, "__self__", None), "__class__", type("", (), {})).__name__
    started_at = time.time()
    _log_preprocess(f"[Data preprocess] {dataset_name}: start total={total}, workers={workers}")
    if workers <= 1 or total <= 1:
        inputs = []
        for i in tqdm(range(total), disable=_disable_preprocess_tqdm()):
            result = preprocess_fn(i)
            if keep_none or result is not None:
                inputs.append(result)
        _log_preprocess(f"[Data preprocess] {dataset_name}: done kept={len(inputs)}/{total}, elapsed={time.time() - started_at:.2f}s")
        return inputs

    inputs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(executor.map(preprocess_fn, range(total)), total=total, disable=_disable_preprocess_tqdm()):
            if keep_none or result is not None:
                inputs.append(result)
    _log_preprocess(f"[Data preprocess] {dataset_name}: done kept={len(inputs)}/{total}, elapsed={time.time() - started_at:.2f}s")
    return inputs


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


def _find_subsequence(sequence, pattern):
    limit = len(sequence) - len(pattern) + 1
    for start in range(limit):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def mask_assistant_response_only(
    tokenizer,
    messages,
    assistant_response,
    max_len=None,
    mask_eos=True,
    enable_thinking=None,
):
    """
    Build labels so that only assistant_response tokens contribute to loss.
    Everything before and after is masked.
    """

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant response")

    chat_template_kwargs = {}
    if enable_thinking is not None:
        chat_template_kwargs["enable_thinking"] = enable_thinking

    # Tokenize the complete conversation. The chat template may normalize the
    # assistant content, especially whitespace around Qwen3 think tags.
    try:
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            **chat_template_kwargs,
        )
    except TypeError:
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
    if isinstance(full_ids, dict):
        full_ids = full_ids["input_ids"]

    # Encode the same conversation without the answer but with the assistant
    # generation header. This is the exact token boundary where the rendered
    # assistant response begins.
    prompt_messages = messages[:-1]
    try:
        prefix_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            **chat_template_kwargs,
        )
    except TypeError:
        prefix_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
    if isinstance(prefix_ids, dict):
        prefix_ids = prefix_ids["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "chat template generation prefix does not match the full conversation"
        )

    total_len = len(full_ids)
    labels = [-100] * total_len
    response_start = len(prefix_ids)

    # Empty responses are used by a few test-mode datasets and must not create
    # trainable labels from template-generated empty think blocks.
    if assistant_response:
        response_end = total_len
        if mask_eos:
            end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            end_start = -1
            if end_ids:
                for start in range(
                    response_start,
                    total_len - len(end_ids) + 1,
                ):
                    if full_ids[start : start + len(end_ids)] == end_ids:
                        end_start = start
                if end_start >= 0:
                    response_end = end_start
        labels[response_start:response_end] = full_ids[
            response_start:response_end
        ]
    elif not mask_eos:
        labels[response_start:] = full_ids[response_start:]

    attention_mask = [1] * total_len

    if max_len is not None and total_len > max_len:
        full_ids = full_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]
        labels = labels[-max_len:]

    return full_ids, attention_mask, labels



class SFTData(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        self.data = _sample_dataframe(self.data, sample, seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        # self.K = K
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        self.get_inputs()  
        _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)
    def __len__(self):
        return len(self.data)


    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        prompt_template = _select_prompt(SFTData_prompts, idx)
        prompt_data = _format_prompt(
            prompt_template,
            inters=history["history_str"],
            item=target_item,
            title=target_item,
            category=self.category,
        )

        prompt = prompt_data["instruction"]

        assistant_response = prompt_data["response"] if not self.test else ""

        messages = _build_messages(prompt, assistant_response)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) >= self.max_len:
                # print(len(input_ids))
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    

    
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=True)
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]
 

class SidSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        self.data = _sample_dataframe(self.data, sample, seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        self.get_inputs()  
        _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)
    
    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        history_str = ", ".join(row["history_item_sid"])
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"Here are the user’s historical interactions: {history}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order.",
                "output": target_item.strip(),
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        prompt_template = _select_prompt(SidSFTDataset_prompts, idx)
        prompt_data = _format_prompt(
            prompt_template,
            inters=history["history_str"],
            item=target_item,
            item_sid=target_item,
            category=self.category,
        )

        prompt = prompt_data["instruction"]

        assistant_response = prompt_data["response"].strip() if not self.test else ""
        if assistant_response and target_item:
            bounded_target_item = _wrap_sid_with_boundary(target_item)
            if target_item in assistant_response:
                assistant_response = assistant_response.replace(target_item, bounded_target_item, 1)
            else:
                assistant_response = bounded_target_item
        messages = _build_messages(prompt, assistant_response)

        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
        )


        if len(input_ids) >= self.max_len:
            print(len(input_ids))
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result


    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=True)
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class EvalSidDataset(Dataset):

    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
        with_title: bool = False,
        sid_metadata_path: str = "",
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        self.data = _sample_dataframe(self.data, sample, seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        self.with_title = with_title
        self.sid_metadata_path = sid_metadata_path
        self.sid2metadata = self._load_sid_metadata() if self.with_title else {}
        self.get_inputs()  
        _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)


    def __len__(self):
        return len(self.data)

    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def _load_sid_metadata(self):
        sid_metadata_path = self.sid_metadata_path
        if not sid_metadata_path:
            sid_metadata_path = f"datasets/index/{self.category}.sid2metadata.json"
        if not os.path.exists(sid_metadata_path):
            raise FileNotFoundError(
                f"with_title=True requires a valid sid_metadata_path, got: {sid_metadata_path}"
            )
        with open(sid_metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _format_sid_with_title(self, sid):
        title = self.sid2metadata.get(sid, {}).get("title", "")
        if title:
            return f'{sid} (title: "{title}")'
        return f'{sid} (title: "")'

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid']) #row['history_item_sid'] #eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        if self.with_title:
            history_entries = [self._format_sid_with_title(str(sid)) for sid in row["history_item_sid"]]
        else:
            history_entries = [str(sid) for sid in row["history_item_sid"]]
        history_str = ", ".join(history_entries)
        for i in range(L):
            if i == 0:
                history += history_entries[i]
            else:
                history += ", " + history_entries[i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"Here is the item interaction history of the user: {history}, what to recommend to the user next?",
                "output": target_item.strip(),
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}
    
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        prompt_template = SidSFTDataset_prompts[0] #_select_prompt(SidSFTDataset_prompts, idx)
        prompt_data = _format_prompt(
            prompt_template,
            inters=history["history_str"],
            item=target_item,
            item_sid=target_item,
            category=self.category,
        )

        prompt = prompt_data["instruction"]
        assistant_response = (
            _wrap_all_sids_with_boundary(prompt_data["response"].strip())
            if not self.test
            else ""
        )
        messages = _build_messages(
            prompt,
            assistant_response,
            include_assistant=not self.test,
        )

        if self.test:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            attention_mask = [1] * len(tokenized)
            prefix_prompt = "<think>\n</think>\n\n"
            prefix_prompt_ids = self.tokenizer.encode(prefix_prompt)
            tokenized = tokenized + prefix_prompt_ids
            attention_mask = attention_mask + [1] * len(prefix_prompt_ids)

            len_prompt = self.max_len + len(prefix_prompt_ids)

            if len(tokenized) >= len_prompt:
                print(len(tokenized))
                tokenized = tokenized[-len_prompt:]
                attention_mask = attention_mask[-len_prompt:]
            return {
                "input_ids": tokenized,
                "attention_mask": attention_mask,
            }

        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=True)
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class SidItemFeatDataset(Dataset):
    def __init__(
        self,
        item_file,
        index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        task_type=None, # select from ['sid2title', 'title2sid'] to indicate the task type, set to None for both
        mask_assistant: bool = True,
    ):
        """
        Dataset for sid2title and title2sid tasks.
        
        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
        """
        random.seed(seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.system_prompt = "You are a helpful assistant for mapping between item titles and semantic IDs."
        self.instruction_text = "Answer the question about item identification."
        self.mask_assistant = mask_assistant
        
        # Build sid2title and title2sid mappings
        self.sid2title = {}
        self.title2sid = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        if task_type is None or task_type == 'sid2title':
            for sid, title in self.sid2title.items():
                self.data.append({
                    'task': 'sid2title',
                    'input': sid,
                    'output': title
                })
        
        # Create title2sid samples
        if task_type is None or task_type == 'title2sid':
            for title, sid in self.title2sid.items():
                self.data.append({
                    'task': 'title2sid',
                    'input': title,
                    'output': sid
                })
        
        self.data = _sample_list(self.data, sample)
        
        if self.tokenizer is not None:
            self.get_inputs()
            _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point, idx=0):
        task_prompts = SidItemFeatDataset_prompts[data_point["task"]]
        prompt_template = _select_prompt(task_prompts, idx)
        if data_point["task"] == "title2sid":
            prompt_data = _format_prompt(
                prompt_template,
                title=data_point["input"],
                item=data_point["output"],
                item_sid=data_point["output"],
                category=self.category,
            )
        else:  # sid2title
            prompt_data = _format_prompt(
                prompt_template,
                item=data_point["input"],
                item_sid=data_point["input"],
                title=data_point["output"],
                category=self.category,
            )
        return prompt_data
    
    def pre(self, idx):
        if self.tokenizer is None:
            return self.data[idx]

        data_point = self.data[idx]
        prompt_data = self.generate_prompt(data_point, idx)
        user_prompt = prompt_data["instruction"]
        assistant_response = prompt_data["response"] if not self.test else ""
        if data_point["task"] == "title2sid" and assistant_response:
            assistant_response = _wrap_all_sids_with_boundary(assistant_response)

        messages = _build_messages(user_prompt, assistant_response)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) > self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=True)
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else [self.pre(i) for i in range(len(self))]
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)

class FusionSeqRecDataset(Dataset):
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(train_file)
        self.data = _sample_dataframe(self.data, sample, seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                # Process description according to requirements:
                # 1. If description is empty, use title
                # 2. If description is a list, select the longest one
                # 3. If the longest in list is also empty, use title
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        # print("self.sid2title: ", self.sid2title)
        # print("self.sid2description: ", self.sid2description)
        self.get_inputs()
        _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)
    
    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title
        
        Args:
            description: The description field from item_feat
            title: The title field from item_feat
        
        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item?"
    
    def generate_prompt_description(self, history):
        return f"Please review the user's historical interactions: {history}, and describe what kind of item he still needs."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
            
        if target_sid in self.sid2description:
            target_description = self.sid2description[target_sid]
            # Clean description if it's a string representation of a list
            if isinstance(target_description, str) and target_description.startswith("['") and target_description.endswith("']"):
                try:
                    desc_list = eval(target_description)
                    target_description = desc_list[0] if desc_list else target_description
                except:
                    pass  # Keep original if eval fails
        else:
            target_description = f"An item with semantic ID {target_sid}"
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_description": target_description,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        """if random.random() < 0.5:
            # Title task
            prompt = self.generate_prompt_title(history_data['history_str'])
            target = history_data['target_title'] + '\n'
        else:
            # Description task
            prompt = self.generate_prompt_description(history_data['history_str'])
            target = history_data['target_description'] + '\n'
        """
        prompt_template = _select_prompt(FusionSeqRecDataset_prompts, idx)
        prompt_data = _format_prompt(
            prompt_template,
            inters=history_data["history_str"],
            title=history_data["target_title"],
            item=history_data["target_title"],
            description=history_data["target_description"],
            item_sid=history_data["target_sid"],
            category=self.category,
        )
        prompt = prompt_data["instruction"]
        target = prompt_data["response"]
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = target if not self.test else ""

        messages = _build_messages(formatted_prompt, assistant_response)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) >= self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=False)
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)


class TitleHistory2SidSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """
        SFT dataset that uses item titles in user history to predict next item's semantic ID.
        
        Args:
            train_file: Path to CSV file with sequence data (must have history_item_title and item_id columns)
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(train_file)
        self.data = _sample_dataframe(self.data, sample, seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        
        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.id2sid[item_id] = combined_sid
        
        self.get_inputs()
        _print_dataset_sample(self.__class__.__name__, self.tokenizer, self.inputs[0] if self.inputs else None)
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""
    
    def get_history(self, row):
        """Extract user history from title sequence and target semantic ID"""
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])
        
        # Format title sequence for prompt
        history_titles = ", ".join([f'"{title}"' for title in history_item_title])
        
        # Get target item's semantic ID from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2sid:
            target_sid = self.id2sid[target_item_id]
        else:
            target_sid = target_item_id  # Fallback to item_id if no semantic ID found
        
        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                last_history_item_id = str(history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False
        
        return {
            "input": history_titles, #f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid,
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }
    
    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        target_output = history_data['output']
        prompt_template = _select_prompt(TitleHistory2SidSFTDataset_prompts, idx)
        prompt_data = _format_prompt(
            prompt_template,
            inter_titles=history_data["history_titles"],
            inters=history_data["history_titles"],
            item=target_output,
            item_sid=target_output,
            category=self.category,
        )
        prompt = prompt_data["instruction"]

        assistant_response = (
            _wrap_all_sids_with_boundary(prompt_data["response"])
            if not self.test
            else ""
        )
        messages = _build_messages(prompt, assistant_response)

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.mask_assistant:
            input_ids, attention_mask, labels = mask_assistant_response_only(
                tokenizer=self.tokenizer,
                messages=messages,
                assistant_response=assistant_response,
                max_len=self.max_len,
            )
        else:
            input_ids = tokenized
            labels = list(tokenized)
            if len(input_ids) >= self.max_len:
                print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
                input_ids = input_ids[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
            else:
                attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=False)
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        result = self.pre(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}


def _load_reasoning_jsonl(path, sample=-1, seed=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    data = pd.DataFrame(rows)
    return _sample_dataframe(data, sample, seed)


def _normalize_sid_history(value):
    if isinstance(value, list):
        return [str(sid) for sid in value]
    if pd.isna(value):
        return []
    try:
        parsed = eval(str(value))
        if isinstance(parsed, list):
            return [str(sid) for sid in parsed]
    except Exception:
        pass
    return [sid for sid in str(value).split() if sid]


def _strip_existing_think_tags(reasoning):
    if isinstance(reasoning, list):
        if len(reasoning) != 1:
            raise ValueError("Expected one reasoning rollout per sample; got %d" % len(reasoning))
        reasoning = reasoning[0]
    if reasoning is None or (isinstance(reasoning, float) and pd.isna(reasoning)):
        return ""
    if not isinstance(reasoning, str):
        raise TypeError("reasoning must be a string or a singleton list of strings")
    reasoning = reasoning.strip()
    if reasoning.startswith("<think>"):
        if "</think>" not in reasoning:
            return ""
        reasoning = reasoning[len("<think>") :].split("</think>", 1)[0].strip()
    return reasoning


def _format_reasoning_user_prompt(history):
    """Canonical Student prompt shared by reasoning training and evaluation."""
    return (
        f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? "
        "Let's think step by step before making recommendation. When reasoning, output only <Step>...</Step> "
        "blocks. After thinking, directly output the item semantic IDs."
    )


SID_BEGIN_TOKEN = "<|sid_begin|>"
SID_END_TOKEN = "<|sid_end|>"
_SID_PATTERN = r"<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?"
_SID_WITH_OPTIONAL_BOUNDARY_RE = re.compile(
    rf"(?:{re.escape(SID_BEGIN_TOKEN)})?({_SID_PATTERN})(?:{re.escape(SID_END_TOKEN)})?"
)


def _wrap_sid_with_boundary(sid):
    sid = str(sid).strip()
    if sid.startswith(SID_BEGIN_TOKEN) and sid.endswith(SID_END_TOKEN):
        return sid
    return f"{SID_BEGIN_TOKEN}{sid}{SID_END_TOKEN}"


def _wrap_all_sids_with_boundary(text):
    """Wrap every SID in label text exactly once."""
    text = str(text)
    return _SID_WITH_OPTIONAL_BOUNDARY_RE.sub(
        lambda match: _wrap_sid_with_boundary(match.group(1)),
        text,
    )


def _format_reasoning_response(reasoning, target_sid):
    wrapped_sid = _wrap_sid_with_boundary(target_sid)
    reasoning = str(reasoning).strip()
    if reasoning:
        return (
            f"<think>\n{reasoning}\n{ASSISTANT_PREFIX}\n"
            f"{wrapped_sid}"
        )
    return (
        f"<think>\n{ASSISTANT_PREFIX}\n"
        f"{wrapped_sid}"
    )


def _tokenize_reasoning_train(
    tokenizer,
    user_prompt,
    assistant_response,
    max_len,
    reasoning=None,
    target_sid=None,
    return_format_mask=False,
):
    messages = _build_messages(user_prompt, assistant_response)
    input_ids, attention_mask, labels = mask_assistant_response_only(
        tokenizer=tokenizer,
        messages=messages,
        assistant_response=assistant_response,
        max_len=None,
    )
    format_label_mask = [0] * len(input_ids)

    if return_format_mask:
        format_tags = ("<think>", "</think>", "<Step>", "</Step>")
        offsets_available = False
        try:
            raw_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            encoded = tokenizer(
                raw_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
            offsets_available = len(offsets) == len(input_ids)
        except (KeyError, TypeError, ValueError, NotImplementedError):
            offsets_available = False

        if offsets_available:
            for tag in format_tags:
                search_start = 0
                while True:
                    char_start = raw_text.find(tag, search_start)
                    if char_start < 0:
                        break
                    char_end = char_start + len(tag)
                    for pos, (token_start, token_end) in enumerate(offsets):
                        overlaps = token_end > char_start and token_start < char_end
                        if overlaps and labels[pos] != -100:
                            format_label_mask[pos] = 1
                    search_start = char_end
        else:
            # Fast-tokenizer offsets are preferred because BPE can merge a tag
            # boundary with adjacent text. This fallback covers tokenizers that
            # expose no offsets when the tag happens to remain a clean sequence.
            for tag in format_tags:
                tag_ids = tokenizer.encode(tag, add_special_tokens=False)
                search_offset = 0
                while tag_ids and search_offset <= len(input_ids) - len(tag_ids):
                    relative_start = _find_subsequence(input_ids[search_offset:], tag_ids)
                    if relative_start < 0:
                        break
                    start = search_offset + relative_start
                    end = start + len(tag_ids)
                    if all(label != -100 for label in labels[start:end]):
                        for pos in range(start, end):
                            format_label_mask[pos] = 1
                    search_offset = start + 1


    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]
        labels = labels[-max_len:]
        format_label_mask = format_label_mask[-max_len:]
    if return_format_mask:
        return input_ids, attention_mask, labels, format_label_mask
    return input_ids, attention_mask, labels


def _tokenize_reasoning_prompt(tokenizer, user_prompt, max_len):
    messages = _build_messages(
        user_prompt,
        include_assistant=False,
    )
    # Render the chat template before tokenization to avoid adding extra tokens.
    raw_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    attention_mask = [1] * len(input_ids)
    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]
    return input_ids, attention_mask


# This dataset is used for reasoning activation task.
# The learning objective is to generate reasoning and answer given user history.

def _load_sid_metadata(item_file):
    """Load SID -> metadata mapping when available.

    The preferred file format is `{sid: metadata}` such as
    `Industrial_and_Scientific.sid2metadata.json`. If another metadata file is
    passed, this function falls back gracefully and the mixed dataset can still
    build SID-only prompts.
    """
    if not item_file or not os.path.exists(item_file):
        return {}
    try:
        with open(item_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    sid2meta = {}
    for key, value in data.items():
        if isinstance(value, dict):
            sid = str(value.get("sid") or key)
            sid2meta[sid] = value
    return sid2meta


def _compact_meta_value(value, max_chars=240):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(x) for x in value if str(x).strip())
    elif isinstance(value, dict):
        pairs = []
        for k, v in list(value.items())[:8]:
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            pairs.append(f"{k}: {v}")
        value = "; ".join(pairs)
    else:
        value = str(value)
    value = " ".join(value.split())
    if len(value) > max_chars:
        return value[:max_chars].rstrip() + "..."
    return value

class ReasoningActivationDataset(Dataset):
    def __init__(
        self,
        reasoning_train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        random.seed(seed)
        self.data = _load_reasoning_jsonl(reasoning_train_file, sample, seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.get_inputs()

    def __len__(self):
        return len(self.inputs) if hasattr(self, "inputs") else len(self.data)

    def generate_prompt_title(self, history):
        return _format_reasoning_user_prompt(history)

    def get_history(self, row):
        history_item_sid = _normalize_sid_history(row["history_item_sid"])
        history_str = ", ".join(history_item_sid)
        target_sid = str(row["item_sid"])
        reasoning = row.get("reasoning", "")
        reasoning = _strip_existing_think_tags(reasoning)
        if not reasoning:
            return None

        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid

        return {
            "history_item_sid": history_item_sid,
            "history_str": history_str,
            "target_sid": target_sid,
            "output": _wrap_sid_with_boundary(target_sid),
            "dedup": is_duplicate,
            "reasoning": reasoning,
            "assistant_response": _format_reasoning_response(reasoning, target_sid),
        }

    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx])
        if history_data is None:
            return None
        if self.dedup and history_data["dedup"]:
            return None

        user_prompt = self.generate_prompt_title(history_data["history_str"])
        assistant_response = history_data["assistant_response"]
        tokenized_result = _tokenize_reasoning_train(
            tokenizer=self.tokenizer,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            max_len=self.max_len,
            reasoning=history_data["reasoning"],
            target_sid=history_data["target_sid"],
        )
        input_ids, attention_mask, labels = tokenized_result

        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result

    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=False)

    def get_inputs_list(self):
        return self.inputs if hasattr(self, "inputs") else []

    def __getitem__(self, idx):
        if hasattr(self, "inputs"):
            return self.inputs[idx]
        return self.pre(idx)




class GeneralSFTReasonDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = []
        with open(train_file, 'r') as f:
            for line in f:
                data_sample = json.loads(line.strip())
                self.data.append(eval(data_sample["messages"]))
        random.seed(seed)
        self.data = _sample_list(self.data, sample)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.cnt = 0
        self.get_inputs()  
        
    
    def __len__(self):
        return len(self.inputs)

    
    def pre(self, idx):
        prompt_messages = []
        for message in self.data[idx]:
            # print(f"message: {message['content']}")
            # message["content"] = eval(message["content"])
            if message["role"] == "user":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "system":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "system", "content": message["content"]})
        try:
            processed_template = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Error processing messages: {self.data[idx]}")
            print(f"Error processing idx {idx}: {e}")
            raise e
        try:
            input_ids = self.tokenizer.encode(processed_template)
        except Exception as e:
            return None
        
        if idx == 0:
            print(f"General data example: {[processed_template]}")
        
        attention_mask = [1] * len(input_ids)

        if self.test:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        golden_output = ""
        for elm in self.data[idx]:
            if elm["role"] == "assistant":
                if type(elm["content"]) == list:
                    golden_output = elm["content"][0]["text"]
                else:
                    golden_output = elm["content"]

        if golden_output == "":
            print(f"No assistant response found in idx {idx}")
            return None
        
        if idx == 0:
            print(f"General data golden output example: {[golden_output]}")
        # golden_output = sample["messages"]
        try:
            golden_tokens = self.tokenizer.encode(golden_output)
        except Exception as e:
            return None
        
        golden_tokens = golden_tokens + [self.tokenizer.eos_token_id]
        
        input_prompt_len = len(input_ids)
        input_ids = input_ids + golden_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100] * input_prompt_len + input_ids[input_prompt_len:]
        
        # if len(input_ids) >= self.max_len:
        #     print(len(input_ids))

        return {
            "input_ids": input_ids[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def get_inputs(self):
        self.inputs = _collect_preprocessed_inputs(self.pre, len(self.data), keep_none=False)
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]



def _safe_literal_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (SyntaxError, ValueError):
        return []


def _format_mapping_lines(inputs, outputs):
    return "\n".join(f"{idx + 1}. {src} -> {dst}" for idx, (src, dst) in enumerate(zip(inputs, outputs)))
