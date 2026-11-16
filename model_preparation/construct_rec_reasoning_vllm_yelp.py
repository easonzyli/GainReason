import argparse
import ast
import json
from pathlib import Path

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


PROMPT_TEMPLATE = (
    "You are an expert in reasoning about user preferences for local business recommendation on Yelp. "
    "Given a user's chronological interaction history, infer the user's current interests and plausible "
    "future interaction directions through concise, evidence-grounded reasoning.\n\n"

    "=== USER INTERACTION HISTORY ===\n"
    "{history_metadata}\n\n"

    "=== SUBSEQUENT INTERACTION (ANSWER; MUST NOT APPEAR EXPLICITLY IN THE REASONING) ===\n"
    "{target_metadata}\n\n"

    "The subsequent interaction is provided only during rationale construction. "
    "Write the reasoning from the perspective that it has not happened yet and the user's actual next interaction is unknown. "
    "Never mention, acknowledge, compare against, or explicitly describe the subsequent interaction. "
    "It represents only one possible realized future, not the uniquely correct outcome.\n\n"

    "The following three stages are a reasoning reference rather than a mandatory checklist. "
    "Use only the stages that are supported by the available evidence, and reduce the number of <Step> blocks when "
    "the history is too sparse to justify deeper reasoning. Do not invent conclusions merely to complete all three stages.\n\n"

    "Stage 1 — Interaction Preference Summary:\n"
    "Summarize the strongest signals in the interaction history, such as recent focus, repeated business categories, "
    "cuisine or service preferences, geographic patterns, price or ambience tendencies, meaningful business relations, "
    "or changes in interest. A single historical interaction does not establish a stable user preference; in sparse histories, "
    "describe only directly supported tendencies and avoid treating incidental business attributes as preferences.\n\n"

    "Stage 2 — Plausible Possibility Expansion:\n"
    "When supported by the history, explore several plausible next-interaction directions, such as continuing a recent category, "
    "revisiting a familiar business type, exploring related cuisines or services, choosing businesses with similar location, price, "
    "ambience, or service characteristics, or shifting toward another emerging interest. "
    "Every proposed direction must be independently supported by the interaction history rather than introduced merely because it "
    "matches the subsequent interaction. Reason divergently before committing to one direction, but do not create unnecessary possibilities.\n\n"

    "Stage 3 — Evidence-Weighted Next-Interaction Inference:\n"
    "When sufficient evidence is available, compare the plausible directions using repetition, recency, specificity, "
    "category consistency, geographic continuity, coherent business attributes, and required assumptions. "
    "Infer the most plausible forward-looking preference or business-interaction direction while retaining uncertainty when "
    "the evidence is weak. Use this stage only when the history provides enough evidence to meaningfully rank competing directions. "
    "Do not predict an exact business.\n\n"

    "Requirements:\n"
    "- Write in English only and output only <Step>...</Step> blocks.\n"
    "- Use one to three <Step> blocks according to the amount and strength of useful evidence.\n"
    "- With only one historical interaction, normally use one or two Steps unless there is sufficient evidence for a meaningful comparison.\n"
    "- Never mention the subsequent interaction, answer, target, or the fact that future information was provided.\n"
    "- Never reveal its business name, exact location, categories, attributes, or other uniquely identifying details.\n"
    "- Every inferred pattern, preference, and possibility must be supportable from the history alone.\n"
    "- Do not retrospectively reinterpret the history merely to make the actual future interaction appear likely or inevitable.\n"
    "- Do not invent unsupported user scenarios, motivations, stable preferences, geographic habits, or temporal patterns.\n"
    "- Keep each Step concise, case-specific, and information-dense.\n\n"

    "Output format:\n"
    "<Step>...</Step>\n"
    "..."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/Qwen3-8B")
    parser.add_argument(
        "--sid_metadata_path",
        default="datasets/index/Yelp.sid2metadata.json",
    )
    parser.add_argument(
        "--train_csv",
        default="datasets/train/Yelp_5_2016-10-2018-11.csv",
    )
    parser.add_argument(
        "--output_jsonl",
        default="datasets/rec_reasoning_verl/Yelp/train_reasoning_4step.jsonl",
    )
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_history", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=496)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser.parse_args()


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except (SyntaxError, ValueError):
        pass
    return [part for part in str(value).split() if part]


def clean_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def parse_nested_value(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


def flatten_mapping(value, prefix=""):
    value = parse_nested_value(value)
    flattened = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.extend(flatten_mapping(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flattened.extend(flatten_mapping(child, child_prefix))
    else:
        text = "None" if value is None else clean_scalar(value)
        flattened.append(f"{prefix}: {text}" if prefix else text)
    return flattened


def join_field(value):
    value = parse_nested_value(value)
    if isinstance(value, dict):
        return "; ".join(flatten_mapping(value))
    if isinstance(value, (list, tuple)):
        return "; ".join(clean_scalar(item) for item in value if clean_scalar(item))
    return clean_scalar(value)


def compact_yelp_metadata(sid, metadata):
    if not metadata:
        return "metadata unavailable"

    details = metadata.get("details") or {}
    attributes = details.get("attributes") or {}
    fields = [
        ("title", metadata.get("title")),
        ("category", metadata.get("categories", metadata.get("category"))),
        ("features", metadata.get("features")),
        ("address", details.get("address")),
        ("state", details.get("state")),
        ("attributes", attributes),
    ]

    parts = []
    for name, value in fields:
        text = join_field(value)
        if text:
            parts.append(f"{name}: {text}")
    return "; ".join(parts)


def build_history_metadata(history_sids, sid2metadata, max_history):
    selected = history_sids[-max_history:]
    lines = [
        f"{index}. {compact_yelp_metadata(sid, sid2metadata.get(sid))}"
        for index, sid in enumerate(selected, 1)
    ]
    return "\n".join(lines) if lines else "(empty history)"


def build_messages(row, sid2metadata, max_history):
    history_sids = parse_list(row.history_item_sid)
    target_sid = str(row.item_sid)
    user_prompt = PROMPT_TEMPLATE.format(
        history_metadata=build_history_metadata(history_sids, sid2metadata, max_history),
        target_metadata=compact_yelp_metadata(target_sid, sid2metadata.get(target_sid)),
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


def main():
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be nonnegative")
    if args.sample < -1:
        raise ValueError("--sample must be -1 or nonnegative")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.tensor_parallel_size <= 0:
        raise ValueError("batch size, max new tokens and tensor parallel size must be positive")
    if args.max_history <= 0:
        raise ValueError("--max_history must be positive")
    output_path = Path(args.output_jsonl)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}; choose a new --output_jsonl path")
    if LLM is None or SamplingParams is None:
        raise ImportError("vLLM is required for construct_rec_reasoning_vllm_yelp.py") from VLLM_IMPORT_ERROR

    with open(args.sid_metadata_path, "r", encoding="utf-8") as fin:
        sid2metadata = json.load(fin)

    dataframe = pd.read_csv(args.train_csv)
    end = len(dataframe) if args.sample < 0 else min(len(dataframe), args.start + args.sample)
    dataframe = dataframe.iloc[args.start:end].reset_index(drop=False).rename(columns={"index": "source_index"})

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
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as fout:
        for offset in tqdm(range(0, len(dataframe), args.batch_size), desc="generate Yelp reasoning"):
            batch_df = dataframe.iloc[offset : offset + args.batch_size]
            messages_batch = [
                build_messages(row, sid2metadata, args.max_history)
                for row in batch_df.itertuples(index=False)
            ]
            model_inputs = build_model_inputs(tokenizer, messages_batch)
            outputs = llm.generate(model_inputs, sampling_params, use_tqdm=False)
            if len(outputs) != len(batch_df):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch_df)} input rows")
            for output in outputs:
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected one reasoning completion per input row")
                if not output.outputs[0].text.strip():
                    raise ValueError("vLLM returned an empty reasoning completion")

            for row, output in zip(batch_df.itertuples(index=False), outputs, strict=True):
                record = {
                    "source_index": int(row.source_index),
                    "user_id": str(row.user_id),
                    "history_item_sid": parse_list(row.history_item_sid),
                    "item_sid": str(row.item_sid),
                    "reasoning": output.outputs[0].text.strip(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"[DONE] wrote {len(dataframe)} records to {output_path}")


if __name__ == "__main__":
    main()
