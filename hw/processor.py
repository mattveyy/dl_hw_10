from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MathVLMProcessor:
    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = self.config.image_size
        resized = image.resize((size, size), Image.BILINEAR)

        arr = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        arr = arr.view(size, size, 3).permute(2, 0, 1).contiguous().float() / 255.0
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        normalized = (arr - mean) / std

        num_tiles = max(1, int(self.config.num_tiles))
        tiles = normalized.unsqueeze(0).expand(num_tiles, -1, -1, -1).contiguous()
        return tiles

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        image_block = (
            IMAGE_START_TOKEN
            + (" " + IMAGE_TOKEN) * self.config.num_image_tokens
            + " "
            + IMAGE_END_TOKEN
        )
        options_text = "\n".join(sample.options) if sample.options else ""
        prompt = (
            f"{image_block}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            f"Ответ:"
        )
        if include_answer:
            prompt = f"{prompt} {sample.answer}"
        return prompt

    def _tokenize(self, text: str) -> list[int]:
        out = self.tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict):
            return list(out["input_ids"])
        return list(out)

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        prompt = self.build_prompt(sample, include_answer=False)
        full = self.build_prompt(sample, include_answer=True)

        prompt_ids = self._tokenize(prompt)
        full_ids = self._tokenize(full)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            full_ids = full_ids + [int(eos_id)]

        max_len = self.config.max_length
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        if len(prompt_ids) > max_len:
            prompt_ids = prompt_ids[:max_len]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        labels = torch.full_like(input_ids, self.config.ignore_index)
        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len < len(full_ids):
            labels[prompt_len:] = input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        ignore = self.config.ignore_index

        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), ignore, dtype=torch.long)

        for i, item in enumerate(batch):
            L = item["input_ids"].shape[0]
            input_ids[i, :L] = item["input_ids"]
            attention_mask[i, :L] = item["attention_mask"]
            labels[i, :L] = item["labels"]

        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
