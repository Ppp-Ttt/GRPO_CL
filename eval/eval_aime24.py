import argparse
import ast
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import process_map
from vllm import LLM, SamplingParams

from math_dapo import compute_score


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


def verify(arg):
    rsp, reward_model = arg
    return compute_score(rsp, reward_model["ground_truth"])


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
    df["res"] = process_map(verify, df[["output", "reward_model"]].values, max_workers=50, chunksize=1)

    timestamp = time.strftime("%m%d_%H%M", time.localtime())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_aime24_{timestamp}.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved: {out_path}")

    score = 0
    for _, row in df.iterrows():
        score += row["res"]["acc"]
    avg_score = score / len(df)
    print(f"acc/mean@32: {avg_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="/code/verl_learning/base_models/Qwen3-8B")
    parser.add_argument("--test_file", type=str, default="/code/verl_learning/data/aime-2024.parquet")
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    print(args)
    main(args)
