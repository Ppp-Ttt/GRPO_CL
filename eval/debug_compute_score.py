import argparse
import json
from pathlib import Path

from math_dapo import compute_score


def load_jsonl_line(path: Path, line_number: int) -> dict:
    if line_number < 1:
        raise ValueError(f"line_number must be >= 1, got {line_number}")

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx == line_number:
                line = line.strip()
                if not line:
                    raise ValueError(f"line {line_number} is empty")
                return json.loads(line)

    raise ValueError(f"line {line_number} out of range")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug helper: read one decoded sample from jsonl and run compute_score."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="/code/verl_learning/eval_results/Qwen3-8B_thinking_aime24/eval_aime24_0411_1353.jsonl",
        help="Path to decode jsonl file.",
    )
    parser.add_argument(
        "--line_number",
        type=int,
        default=1,
        help="1-based line number in jsonl.",
    )
    parser.add_argument(
        "--strict_box_verify",
        action="store_true",
        help="Use strict box verification mode in compute_score.",
    )
    args = parser.parse_args()

    path = Path(args.jsonl)
    row = load_jsonl_line(path, args.line_number)

    output = row.get("output", "")
    reward_model = row.get("reward_model", {})
    ground_truth = reward_model.get("ground_truth")

    if ground_truth is None:
        raise ValueError("reward_model.ground_truth not found in selected line")
    args.strict_box_verify=False
    result = compute_score(
        solution_str=output,
        ground_truth=ground_truth,
        strict_box_verify=args.strict_box_verify,
    )

    print(f"file: {path}")
    print(f"line_number: {args.line_number}")
    print(f"data_source: {row.get('data_source')}")
    print(f"ground_truth: {ground_truth}")
    print("score_result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
