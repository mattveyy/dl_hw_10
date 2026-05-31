from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    if not text:
        return None
    upper = text.upper()
    allowed = "".join(choices)

    explicit = re.search(rf"(?:ANSWER|ОТВЕТ)\s*[:\-]?\s*\(?\s*([{allowed}])\b", upper)
    if explicit:
        return explicit.group(1)

    paren = re.search(rf"\(\s*([{allowed}])\s*\)", upper)
    if paren:
        return paren.group(1)

    pattern = re.compile(rf"(?:^|[^A-Z0-9А-Я])([{allowed}])(?:[^A-Z0-9А-Я]|$)")
    match = pattern.search(upper)
    if match:
        return match.group(1)
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    from hw.dataset import MathVQADataset

    data_cfg = config.get("data", {})
    manifest = data_cfg.get("eval_manifest") or data_cfg.get("train_manifest")
    split = data_cfg.get("split", "dev")
    max_samples = data_cfg.get("max_samples")

    dataset = MathVQADataset(manifest_path=manifest, split=split, max_samples=max_samples)

    predictions: list[dict[str, Any]] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = build_benchmark_prompt(sample.question, sample.options)
        fake_output = sample.options[0] if sample.options else "A"
        pred = parse_mc_answer(fake_output) or "A"
        predictions.append(
            {
                "id": sample.id,
                "prompt": prompt,
                "prediction": pred,
                "answer": sample.answer,
                "subject": sample.subject,
            }
        )

    out_path = config.get("inference", {}).get("output_path")
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
