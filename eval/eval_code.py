import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import process_map
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERL_PACKAGE_ROOT = PROJECT_ROOT / "verl"
if str(VERL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_PACKAGE_ROOT))

from verl.utils.reward_score.prime_code import compute_score  # noqa: E402


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_messages(prompt_obj: Any) -> list[dict[str, str]]:
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()

    if isinstance(prompt_obj, str):
        try:
            prompt_obj = ast.literal_eval(prompt_obj)
        except Exception:
            return [{"role": "user", "content": prompt_obj}]

    if isinstance(prompt_obj, dict):
        prompt_obj = [prompt_obj]

    if not isinstance(prompt_obj, list):
        return [{"role": "user", "content": str(prompt_obj)}]

    messages: list[dict[str, str]] = []
    for item in prompt_obj:
        if isinstance(item, dict):
            messages.append(
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
            )
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def parse_reward_model(reward_model_obj: Any) -> dict[str, Any]:
    if isinstance(reward_model_obj, dict):
        return reward_model_obj

    if isinstance(reward_model_obj, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(reward_model_obj)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    return {}


def verify(arg: tuple[str, Any]) -> dict[str, Any]:
    rsp, reward_model_obj = arg
    reward_model = parse_reward_model(reward_model_obj)
    ground_truth = reward_model.get("ground_truth")

    if ground_truth is None:
        return {
            "score": 0.0,
            "metadata": None,
            "error": "missing reward_model.ground_truth",
        }

    try:
        score, metadata = compute_score(rsp, ground_truth, continuous=True)
        return {
            "score": float(score),
            "metadata": metadata,
        }
    except Exception as e:
        return {
            "score": 0.0,
            "metadata": None,
            "error": repr(e),
        }


def main(args):
    df = pd.read_parquet(args.test_file)
    messages = [parse_messages(x) for x in df["prompt"]]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )

    start = time.time()
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
    )
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    df["output"] = [o.outputs[0].text if o.outputs else "" for o in outputs]
    df["res"] = process_map(
        verify,
        df[["output", "reward_model"]].values,
        max_workers=args.eval_workers,
        chunksize=1,
    )

    df["score"] = df["res"].apply(lambda x: float(x.get("score", 0.0)) if isinstance(x, dict) else 0.0)
    df["pass"] = df["score"].apply(lambda x: 1.0 if x >= 1.0 else 0.0)

    timestamp = time.strftime("%m%d_%H%M", time.localtime())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_{timestamp}.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved: {out_path}")

    avg_score = df["score"].mean() if len(df) > 0 else 0.0
    pass_rate = df["pass"].mean() if len(df) > 0 else 0.0
    print(f"score/mean@32: {avg_score}")
    print(f"pass@1/mean@32: {pass_rate}")

    source_scores = defaultdict(list)
    source_pass = defaultdict(list)
    if "data_source" in df.columns:
        for _, row in df.iterrows():
            src = str(row["data_source"])
            source_scores[src].append(row["score"])
            source_pass[src].append(row["pass"])
    else:
        for _, row in df.iterrows():
            source_scores["unknown"].append(row["score"])
            source_pass["unknown"].append(row["pass"])

    source_score_means = {
        source: (sum(vals) / len(vals) if vals else 0.0)
        for source, vals in source_scores.items()
    }
    source_pass_means = {
        source: (sum(vals) / len(vals) if vals else 0.0)
        for source, vals in source_pass.items()
    }

    source_level_score = (
        sum(source_score_means.values()) / len(source_score_means) if source_score_means else 0.0
    )
    source_level_pass = (
        sum(source_pass_means.values()) / len(source_pass_means) if source_pass_means else 0.0
    )

    print(f"score_by_data_source/mean@32: {source_level_score}")
    print(f"pass_by_data_source/mean@32: {source_level_pass}")

    log_path = output_dir / f"eval_{timestamp}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"jsonl_file: {out_path}\n")
        f.write(f"num_samples: {len(df)}\n")
        f.write(f"score/mean@32: {avg_score:.6f}\n")
        f.write(f"pass@1/mean@32: {pass_rate:.6f}\n")
        f.write(f"score_by_data_source/mean@32: {source_level_score:.6f}\n")
        f.write(f"pass_by_data_source/mean@32: {source_level_pass:.6f}\n")

        f.write("score_by_data_source:\n")
        for source in sorted(source_scores.keys()):
            vals = source_scores[source]
            f.write(f"  - {source}: {source_score_means[source]:.6f} ({len(vals)} samples)\n")

        f.write("pass_by_data_source:\n")
        for source in sorted(source_pass.keys()):
            vals = source_pass[source]
            f.write(f"  - {source}: {source_pass_means[source]:.6f} ({len(vals)} samples)\n")

    print(f"Saved: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="/code/verl_learning/base_models/Qwen3-8B")
    parser.add_argument("--test_file", type=str, default="/code/verl_learning/data/test/code_test.parquet")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--eval_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    print(args)
    main(args)
