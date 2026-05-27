# coding=utf-8
"""Qwen2.5-VL 多模态编码：文本 + 图像序列 → 定长 hidden（供 Video / World 条件）。"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

os.environ.setdefault("TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, str, bytes]
ImageSequenceInput = Union[ImageInput, Sequence[ImageInput]]


class Qwen25Embedder:
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=512,
        local_files_only=False,
        max_pixels: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if model_kwargs is None:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device,
            }
            if use_offload_folder is not None:
                model_kwargs["offload_folder"] = use_offload_folder

        self.processor = AutoProcessor.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        ).eval()
        self.model.requires_grad_(False)

        self.model_max_length = int(model_max_length)
        tok_max = getattr(self.processor.tokenizer, "model_max_length", None)
        if tok_max is not None and int(tok_max) < self.model_max_length:
            self.model_max_length = int(tok_max)

        _default_max_pixels = 1280 * 28 * 28
        _default_min_pixels = 256 * 28 * 28
        image_proc = getattr(self.processor, "image_processor", None)
        if image_proc is not None:
            if max_pixels is not None:
                image_proc.max_pixels = max_pixels
            elif getattr(image_proc, "max_pixels", None) in (None, 0) or image_proc.max_pixels > _default_max_pixels * 4:
                image_proc.max_pixels = _default_max_pixels
            if getattr(image_proc, "min_pixels", None) in (None, 0):
                image_proc.min_pixels = _default_min_pixels

    @property
    def hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            tc = getattr(cfg, "text_config", None)
            if tc is not None and getattr(tc, "hidden_size", None):
                return int(tc.hidden_size)
            if getattr(cfg, "hidden_size", None):
                return int(cfg.hidden_size)
        return int(self.model.get_input_embeddings().weight.shape[1])

    @staticmethod
    def _load_pil_image(item: ImageInput) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, bytes):
            return Image.open(BytesIO(item)).convert("RGB")
        return Image.open(item).convert("RGB")

    def _coerce_image_list(
        self,
        images: Optional[ImageSequenceInput],
        *,
        max_images: Optional[int] = None,
    ) -> Optional[List[Image.Image]]:
        if images is None:
            return None
        if isinstance(images, (Image.Image, str, bytes)):
            seq: List[ImageInput] = [images]
        else:
            seq = list(images)
        if not seq:
            return None
        if max_images is not None and len(seq) > max_images:
            idx = [
                int(round(i * (len(seq) - 1) / (max_images - 1))) if max_images > 1 else 0
                for i in range(max_images)
            ]
            seq = [seq[i] for i in idx]
        return [self._load_pil_image(im) for im in seq]

    def _messages_from_texts(self, text_or_texts: str | list[str], images: Optional[List[Image.Image]] = None):
        if isinstance(text_or_texts, list):
            return [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in text_or_texts]
        text = text_or_texts
        if images:
            return [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image} for image in images]
                    + [{"type": "text", "text": text}],
                }
            ]
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _pad_hidden_to_max_length(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        warn_truncate: bool = True,
    ) -> torch.Tensor:
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        max_len = int(self.model_max_length)
        if warn_truncate and (seq_lens > max_len).any():
            logger.warning(
                "Qwen2.5 VLM seq_len %s > model_max_length=%s，将截断",
                int(seq_lens.max().item()),
                max_len,
            )
        padded_rows = []
        for i, length in enumerate(seq_lens):
            length = int(length.item())
            row = hidden[i, :length]
            if length < max_len:
                pad = row.new_zeros(max_len - length, row.size(-1))
                row = torch.cat([row, pad], dim=0)
            else:
                row = row[:max_len]
            padded_rows.append(row)
        return torch.stack(padded_rows, dim=0)

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int):
        if isinstance(vlm_inputs, list):
            input_ids_list = [v["input_ids"] for v in vlm_inputs]
            attention_mask_list = [v.get("attention_mask") for v in vlm_inputs]
            pixel_values_list = [v.get("pixel_values") for v in vlm_inputs]
            image_grid_thw_list = [v.get("image_grid_thw") for v in vlm_inputs]

            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            input_ids_batch = vlm_inputs["input_ids"].to(self.device)
            attention_mask_batch = vlm_inputs["attention_mask"].to(self.device)
            pixel_values_batch = vlm_inputs["pixel_values"].to(self.device)
            image_grid_thw_batch = vlm_inputs["image_grid_thw"].to(self.device)

        inputs_embeds = self.model.get_input_embeddings()(input_ids_batch)

        image_out = self.model.get_image_features(pixel_values_batch, image_grid_thw_batch)
        deepstack_image_embeds = None
        if isinstance(image_out, tuple):
            image_embeds = image_out[0]
            if len(image_out) > 1:
                deepstack_image_embeds = image_out[1]
        else:
            image_embeds = image_out

        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.torch_dtype)
        else:
            image_embeds = image_embeds.to(self.device, self.torch_dtype)

        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        position_ids, _ = self.model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,
            attention_mask=attention_mask_batch,
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids

    @torch.no_grad()
    def extract_und_features(self, vlm_inputs) -> torch.Tensor:
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs["input_ids"].shape[0]

        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = (
            self._process_vlm_inputs_to_tokens(vlm_inputs, B)
        )

        vlm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if visual_pos_masks is not None:
            vlm_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs["deepstack_visual_embeds"] = deepstack_image_embeds

        lm = getattr(self.model.model, "language_model", self.model.model)
        vlm_output = lm(**vlm_kwargs)
        return vlm_output.hidden_states[-1]

    def _preprocess_vlm_messages(
        self,
        instruction: str,
        images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, torch.Tensor]:
        messages = self._messages_from_texts(instruction, images)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        proc_images = images if images else None
        encoded = self.processor(text=[text], images=proc_images, return_tensors="pt")
        vlm_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "pixel_values": encoded["pixel_values"].to(self.device),
            "image_grid_thw": encoded.get("image_grid_thw", None),
        }
        if vlm_inputs["image_grid_thw"] is not None:
            vlm_inputs["image_grid_thw"] = vlm_inputs["image_grid_thw"].to(self.device)
        return vlm_inputs

    @torch.no_grad()
    def get_vlm_embeddings(
        self,
        text: str,
        images: Optional[ImageSequenceInput] = None,
        *,
        max_images: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pil_images = self._coerce_image_list(images, max_images=max_images)
        vlm_inputs = self._preprocess_vlm_messages(text, pil_images)
        hidden = self.extract_und_features(vlm_inputs)
        attention_mask = vlm_inputs["attention_mask"]
        embeddings = self._pad_hidden_to_max_length(hidden, attention_mask)
        return embeddings.detach(), attention_mask.detach()

    @torch.no_grad()
    def get_vlm_embeddings_batch(
        self,
        samples: Sequence[Tuple[str, Optional[ImageSequenceInput]]],
        *,
        max_images: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("samples 不能为空")
        if len(samples) == 1:
            text, images = samples[0]
            return self.get_vlm_embeddings(text, images, max_images=max_images)

        vlm_inputs_list = []
        for text, images in samples:
            pil_images = self._coerce_image_list(images, max_images=max_images)
            vlm_inputs_list.append(self._preprocess_vlm_messages(text, pil_images))
        hidden = self.extract_und_features(vlm_inputs_list)
        attention_mask = torch.cat([x["attention_mask"] for x in vlm_inputs_list], dim=0)
        embeddings = self._pad_hidden_to_max_length(hidden, attention_mask)
        return embeddings.detach(), attention_mask.detach()
