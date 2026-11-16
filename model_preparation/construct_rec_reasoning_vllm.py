import argparse
import ast
import json
import random
from pathlib import Path
import os

os.environ["NCCL_P2P_DISABLE"] = "1"

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:
    LLM = None
    SamplingParams = None
    VLLM_IMPORT_ERROR = exc
else:
    VLLM_IMPORT_ERROR = None

PROMPT_TEMPLATES = [
    (
    "You are an expert in reasoning about user shopping behavior for e-commerce recommendation. "
    "Given a user's chronological purchase history, infer the user's current shopping interests and plausible "
    "future needs through concise, evidence-grounded reasoning.\n\n"

    "=== USER PURCHASE HISTORY ===\n"
    "{history_metadata}\n\n"

    "=== SUBSEQUENT INTERACTION (ANSWER; MUST NOT APPEAR EXPLICITLY IN THE REASONING) ===\n"
    "{target_metadata}\n\n"

    "The subsequent interaction is provided only during rationale construction. "
    "Write the reasoning from the perspective that it has not happened yet and the user's actual next purchase is unknown. "
    "Never mention, acknowledge, compare against, or explicitly describe the subsequent interaction. "
    "It represents only one possible realized future, not the uniquely correct outcome.\n\n"

    "The following three stages are a reasoning reference rather than a mandatory checklist. "
    "Use only the stages that are supported by the available evidence, and reduce the number of <Step> blocks when "
    "the history is too sparse to justify deeper reasoning. Do not invent conclusions merely to complete all three stages.\n\n"

    "Step 1 — Shopping Behavior Summary:\n"
    "Summarize the strongest signals in the purchase history, such as recent focus, repetition, shared specifications, "
    "meaningful product relations, or changes in interest. When evidence is sparse, keep the conclusion local rather "
    "than inferring stable preferences or recurring needs.\n\n"

    "Step 2 — Plausible Possibility Expansion:\n"
    "When supported by the history, explore several plausible next-purchase directions, such as continuation, replenishment, "
    "complementarity, substitution, specification change, task continuation, exploration, or interest shift. "
    "Reason divergently before committing to one direction, but do not create unnecessary possibilities.\n\n"

    "Step 3 — Evidence-Weighted Next-Purchase Inference:\n"
    "When sufficient evidence is available, compare the plausible directions using repetition, recency, specificity, "
    "temporal continuity, coherent product relations, and required assumptions. Infer the most plausible forward-looking "
    "shopping need or direction while retaining uncertainty when the evidence is weak. Do not predict an exact product.\n\n"

    "Requirements:\n"
    "- Write in English only and output only <Step>...</Step> blocks.\n"
    "- Use one to three <Step> blocks according to the amount of useful evidence.\n"
    "- Never mention the subsequent interaction, answer, target, or the fact that future information was provided.\n"
    "- Never reveal its title, brand, exact specifications, or other uniquely identifying details.\n"
    "- Do not retrospectively reinterpret the history merely to make the actual future interaction appear inevitable.\n"
    "- Do not invent unsupported user scenarios, motivations, stable preferences, or temporal patterns.\n"
    "- Keep each Step concise, case-specific, and information-dense.\n\n"

    "Output format:\n"
    "<Step>...</Step>\n"
    "..."
)
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/Qwen3-8B/")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--train_parquet", default="")
    parser.add_argument("--sid_metadata_path", default="")
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--output_jsonl", default="")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--rollout_n",
        "--num_rollouts_per_sample",
        dest="rollout_n",
        type=int,
        default=1,
        help="Number of independent reasoning rollouts generated for each input row.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser.parse_args()


def fill_default_paths(args):
    category = args.category
    if not args.train_parquet:
        args.train_parquet = (
            f"datasets/rec_reasoning_verl/{category}/train.parquet"
        )
    if not args.sid_metadata_path:
        args.sid_metadata_path = f"datasets/index/{category}.sid2metadata.json"
    if not args.train_csv:
        args.train_csv = f"datasets/train/{category}_5_2016-10-2018-11.csv"
    if not args.output_jsonl:
        args.output_jsonl = (
            f"datasets/rec_reasoning_verl/{category}/train_reasoning_4step.jsonl"
        )
    return args


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except (SyntaxError, ValueError):
        pass
    return [x for x in str(value).split() if x]


def clean_text(value, max_chars=500):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value if x)
    elif isinstance(value, dict):
        value = "; ".join(f"{k}: {v}" for k, v in value.items() if v)
    else:
        value = str(value)
    value = " ".join(value.replace("\n", " ").split())
    return value[:max_chars]


def compact_metadata(metadata):
    if not metadata:
        return "(metadata unavailable)"
    category = metadata.get("category", metadata.get("categories", ""))
    parts = []
    fields = [
        ("title", metadata.get("title"), 240),
        ("brand", metadata.get("brand"), 120),
        ("category", category, 160),
    ]
    for name, value, max_chars in fields:
        text = clean_text(value, max_chars)
        if text:
            parts.append(f"{name}: {text}")
    return "; ".join(parts) if parts else "(metadata unavailable)"


def build_history_metadata(history_sids, sid2metadata, max_history=10):
    selected = history_sids[-max_history:]
    lines = []
    for idx, sid in enumerate(selected, 1):
        lines.append(f"{idx}. {compact_metadata(sid2metadata.get(sid))}")
    return "\n".join(lines) if lines else "(empty history)"


def build_messages(row, sid2metadata, rng):
    history_sids = parse_list(row.history_item_sid)
    target_sid = str(row.item_sid)
    history_metadata = build_history_metadata(history_sids, sid2metadata)
    target_metadata = compact_metadata(sid2metadata.get(target_sid))
    template = PROMPT_TEMPLATES[0] #rng.choice(PROMPT_TEMPLATES)
    user_prompt = template.format(
        history_metadata=history_metadata,
        target_metadata=target_metadata,
    )
    return [
        {"role": "system", "content": "You are a helpful assistant.\n"},
        {"role": "user", "content": user_prompt},
    ]


def build_model_inputs(tokenizer, message_batch):
    texts = []
    for messages in message_batch:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        texts.append(text)
    return texts


def extract_reasoning_rollouts(output, expected_n, source_index):
    """Return every vLLM completion for one input and enforce exact cardinality."""
    candidates = list(output.outputs)
    if len(candidates) != expected_n:
        raise RuntimeError(
            f"vLLM returned {len(candidates)} rollouts for source_index={source_index}, "
            f"expected {expected_n}"
        )
    texts = [candidate.text.strip() for candidate in candidates]
    if any(not text for text in texts):
        raise ValueError(f"vLLM returned empty reasoning for source_index={source_index}")
    return texts


def _validate_resume_record(record, expected_row, rollout_n, record_number):
    """Validate one saved JSONL record against the selected source row."""
    required = {
        "source_index",
        "user_id",
        "history_item_sid",
        "item_sid",
        "reasoning",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(
            f"Existing output record {record_number} is missing fields: {missing}"
        )

    observed = {
        "source_index": int(record["source_index"]),
        "user_id": str(record["user_id"]),
        "history_item_sid": parse_list(record["history_item_sid"]),
        "item_sid": str(record["item_sid"]),
    }
    expected = {
        "source_index": int(expected_row.source_index),
        "user_id": str(expected_row.user_id),
        "history_item_sid": parse_list(expected_row.history_item_sid),
        "item_sid": str(expected_row.item_sid),
    }
    if observed != expected:
        raise ValueError(
            f"Existing output record {record_number} is not aligned with the current input: "
            f"observed={observed}, expected={expected}"
        )

    reasoning = record["reasoning"]
    if isinstance(reasoning, str):
        observed_n = 1
    elif isinstance(reasoning, list):
        observed_n = len(reasoning)
    else:
        raise TypeError(
            f"Existing output record {record_number} has invalid reasoning type "
            f"{type(reasoning).__name__}"
        )
    if observed_n != rollout_n:
        raise ValueError(
            f"Existing output record {record_number} has {observed_n} reasoning rollouts, "
            f"but --rollout_n={rollout_n}"
        )


def load_resume_count(output_path, selected_df, rollout_n):
    """Return the validated completed prefix length from an existing JSONL.

    A malformed final line is treated as an interrupted write and truncated.
    Malformed internal lines and source-data mismatches fail loudly.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0

    completed = 0
    valid_bytes = 0
    last_valid_line_had_newline = True
    with output_path.open("rb") as fin:
        line_number = 0
        while True:
            raw_line = fin.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                if fin.read().strip():
                    raise ValueError(
                        f"Existing output contains a blank internal line at line {line_number}"
                    )
                break
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if fin.read().strip():
                    raise ValueError(
                        f"Existing output contains malformed JSON before its final line at "
                        f"line {line_number}"
                    ) from error
                print(
                    f"[RESUME] Truncating incomplete final JSONL line {line_number} "
                    f"from {output_path}"
                )
                break

            if completed >= len(selected_df):
                raise ValueError(
                    f"Existing output has more records than the selected input range: "
                    f"at least {completed + 1} > {len(selected_df)}"
                )
            expected_row = selected_df.iloc[completed]
            _validate_resume_record(record, expected_row, rollout_n, completed + 1)
            completed += 1
            valid_bytes = fin.tell()
            last_valid_line_had_newline = raw_line.endswith(b"\n")

    file_size = output_path.stat().st_size
    if valid_bytes != file_size:
        with output_path.open("r+b") as fout:
            fout.truncate(valid_bytes)
    elif valid_bytes and not last_valid_line_had_newline:
        with output_path.open("ab") as fout:
            fout.write(b"\n")

    print(
        f"[RESUME] validated {completed}/{len(selected_df)} completed records "
        f"in {output_path}"
    )
    return completed


def main():
    args = fill_default_paths(parse_args())
    if args.start < 0:
        raise ValueError("--start must be nonnegative")
    if args.sample < -1:
        raise ValueError("--sample must be -1 or nonnegative")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.tensor_parallel_size <= 0:
        raise ValueError("batch size, max new tokens and tensor parallel size must be positive")
    if args.rollout_n <= 0:
        raise ValueError(f"--rollout_n must be positive, got {args.rollout_n}")
    random.seed(args.seed)
    rng = random.Random(args.seed)

    with open(args.sid_metadata_path, "r", encoding="utf-8") as f:
        sid2metadata = json.load(f)

    csv_df = pd.read_csv(args.train_csv)
    # parquet_df = pd.read_parquet(args.train_parquet)
    # if len(csv_df) != len(parquet_df):
    #     print(f"[WARN] CSV rows ({len(csv_df)}) != parquet rows ({len(parquet_df)}). Use CSV rows for generation.")

    end = len(csv_df) if args.sample < 0 else min(len(csv_df), args.start + args.sample)
    csv_df = csv_df.iloc[args.start:end].reset_index(drop=False).rename(columns={"index": "source_index"})

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_count = load_resume_count(output_path, csv_df, args.rollout_n)
    if resume_count == len(csv_df):
        print(
            f"[DONE] output already contains all {len(csv_df)} records and "
            f"{len(csv_df) * args.rollout_n} reasoning rollouts: {output_path}"
        )
        return

    if LLM is None or SamplingParams is None:
        raise ImportError("vLLM is required for construct_rec_reasoning_vllm.py") from VLLM_IMPORT_ERROR

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        n=args.rollout_n,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    generated_count = 0
    with output_path.open("a", encoding="utf-8") as fout, tqdm(
        total=len(csv_df),
        initial=resume_count,
        desc="generate reasoning",
    ) as progress:
        for offset in range(resume_count, len(csv_df), args.batch_size):
            batch_df = csv_df.iloc[offset : offset + args.batch_size]
            messages_batch = [build_messages(row, sid2metadata, rng) for row in batch_df.itertuples(index=False)]
            texts = build_model_inputs(tokenizer, messages_batch)
            outputs = llm.generate(texts, sampling_params, use_tqdm=False)
            if len(outputs) != len(batch_df):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} request outputs for a batch of {len(batch_df)} rows"
                )
            reasoning_lists = [
                extract_reasoning_rollouts(output, args.rollout_n, int(row.source_index))
                for row, output in zip(batch_df.itertuples(index=False), outputs, strict=True)
            ]

            for row, reasoning_list in zip(
                batch_df.itertuples(index=False), reasoning_lists, strict=True
            ):
                history_sids = parse_list(row.history_item_sid)
                target_sid = str(row.item_sid)
                record = {
                    "source_index": int(row.source_index),
                    "user_id": str(row.user_id),
                    "history_item_sid": history_sids,
                    "item_sid": target_sid,
                    "reasoning": reasoning_list[0] if args.rollout_n == 1 else reasoning_list,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                generated_count += 1
                progress.update(1)
            os.fsync(fout.fileno())

    print(
        f"[DONE] output contains {len(csv_df)} records and "
        f"{len(csv_df) * args.rollout_n} reasoning rollouts at {output_path}; "
        f"generated {generated_count} new records in this run"
    )


if __name__ == "__main__":
    main()
