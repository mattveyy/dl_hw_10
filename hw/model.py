from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(vision_hidden_states)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)

        B, L, D = x.shape
        if L != self.num_image_tokens:
            x = x.transpose(1, 2)
            x = torch.nn.functional.adaptive_avg_pool1d(x, self.num_image_tokens)
            x = x.transpose(1, 2)
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    output = input_embeds.clone()
    B = input_ids.shape[0]
    K = visual_embeds.shape[1]
    for b in range(B):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        k = min(positions.numel(), K)
        output[b, positions[:k]] = visual_embeds[b, :k].to(output.dtype)
    return output


class MathVLM(nn.Module):
    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.view(B * T, C, H, W)
        out = self.vision_encoder(flat)
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        _, L, D = hidden.shape
        hidden = hidden.view(B, T * L, D)
        return hidden

    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed_layer = self.language_model.get_input_embeddings()
        return embed_layer(input_ids)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        vision_hidden = self._encode_images(pixel_values)
        visual_embeds = self.adapter(vision_hidden)

        text_embeds = self._text_embeddings(input_ids)
        merged = merge_visual_embeddings(
            text_embeds, input_ids, visual_embeds, self.config.image_token_id
        )

        return self.language_model(
            inputs_embeds=merged,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        vision_hidden = self._encode_images(pixel_values)
        visual_embeds = self.adapter(vision_hidden)
        text_embeds = self._text_embeddings(input_ids)
        merged = merge_visual_embeddings(
            text_embeds, input_ids, visual_embeds, self.config.image_token_id
        )

        return self.language_model.generate(
            inputs_embeds=merged,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
