from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_loss(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["loss"]
    if hasattr(output, "loss"):
        return output.loss
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Cannot extract loss from output of type {type(output)}")


def train_one_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    optimizer.zero_grad()
    output = model(batch)
    loss = _extract_loss(output)
    if not torch.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss.item()}")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    dataset = MathVQADataset(
        manifest_path=data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    processor_config = ProcessorConfig(
        image_size=proc_cfg.get("image_size", 224),
        num_tiles=proc_cfg.get("num_tiles", 1),
        tile_overlap=proc_cfg.get("tile_overlap", 0.0),
        num_image_tokens=proc_cfg.get("num_image_tokens", 49),
        max_length=proc_cfg.get("max_length", 512),
        ignore_index=proc_cfg.get("ignore_index", -100),
    )

    class _ToyTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<image>": 2}

        def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None):
            ids: list[int] = []
            for tok in text.replace("\n", " ").split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                ids.append(self.vocab[tok])
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer = _ToyTokenizer()
    processor = MathVLMProcessor(tokenizer, processor_config)

    samples = [processor(dataset[i]) for i in range(len(dataset))]

    device = torch.device(trainer_cfg.get("device", "cpu"))

    class _ToyVLM(torch.nn.Module):
        def __init__(self, num_image_tokens: int, image_size: int) -> None:
            super().__init__()
            in_dim = 3 * image_size * image_size
            self.visual_proj = torch.nn.Linear(in_dim, 32)
            self.text_embed = torch.nn.Embedding(32000, 32)
            self.head = torch.nn.Linear(32, 32000)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            B, T, C, H, W = pixel_values.shape
            vis = self.visual_proj(pixel_values.view(B, T * C * H * W // T).clamp(-5, 5))
            txt = self.text_embed(input_ids).mean(dim=1)
            mixed = vis + txt
            logits = self.head(mixed)
            target = labels.clamp(min=0)[:, 0]
            loss = torch.nn.functional.cross_entropy(logits, target)
            return {"loss": loss, "logits": logits}

    model = _ToyVLM(processor_config.num_image_tokens, processor_config.image_size).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(trainer_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    grad_accum = max(1, int(trainer_cfg.get("global_batch_size", batch_size)) // batch_size)
    max_steps = int(trainer_cfg.get("max_steps", 3))
    if fast_train:
        max_steps = min(max_steps, 2)

    def _iter_batches():
        i = 0
        while True:
            chunk = []
            for _ in range(batch_size):
                chunk.append(samples[i % len(samples)])
                i += 1
            yield processor.collate(chunk)

    losses: list[float] = []
    optimizer.zero_grad()
    batch_iter = _iter_batches()
    for step in range(max_steps):
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = next(batch_iter)
            model.train()
            output = model(batch)
            loss = _extract_loss(output) / grad_accum
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at step {step}")
            loss.backward()
            step_loss += float(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        losses.append(step_loss)

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter_like": model.state_dict()}, path)

    return {"losses": losses, "num_steps": len(losses)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    result = run_training(config, fast_train=args.fast_train)
    print({"losses": result["losses"], "num_steps": result["num_steps"]})


if __name__ == "__main__":
    main()
