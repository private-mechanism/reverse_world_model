import torch
import os

os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, List, Optional, Sequence, Tuple, Union

ImageInput = Union[Image.Image, str, bytes]
ImageSequenceInput = Union[ImageInput, Sequence[ImageInput]]

class Qwen3Embedder:
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
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
                "device_map": self.device, #"auto",
            }
            if use_offload_folder is not None:
                model_kwargs["offload_folder"] = use_offload_folder
                # 可按需配置 device_map 做部分 offload

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        print("=============== Loading processor ===============")
        self.processor = AutoProcessor.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        print("=============== Loading model ===============")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        ).eval()
        print("=============== Model loaded ===============")
        self.model.requires_grad_(False)
        # 与 umT5 / World 对齐的存盘长度；勿直接用 Qwen tokenizer 的超大 context（常见 262144）
        self.model_max_length = int(model_max_length)
        tok_max = getattr(self.processor.tokenizer, "model_max_length", None)
        if tok_max is not None and int(tok_max) < self.model_max_length:
            self.model_max_length = int(tok_max)

        # 限制视觉 token 数，避免高分辨率图 → 数十万 seq（如 512×512 patch 网格）
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

    @staticmethod
    def _load_pil_image(item: ImageInput) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        if isinstance(item, bytes):
            from io import BytesIO

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
            # 均匀下采样图像序列，避免超长上下文
            idx = [
                int(round(i * (len(seq) - 1) / (max_images - 1))) if max_images > 1 else 0
                for i in range(max_images)
            ]
            seq = [seq[i] for i in idx]
        return [self._load_pil_image(im) for im in seq]

    def _messages_from_texts(self, text_or_texts: str | list[str], images: Optional[List[Image.Image]] = None):
        """支持 (text, image) 或 (texts 列表)。返回一条 conversation 或 list of conversations。"""
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
            import logging

            logging.getLogger(__name__).warning(
                "Qwen3 VLM seq_len %s > model_max_length=%s，将截断；可减小输入分辨率或 --qwen3-num-frames",
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

    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of (inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids)
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)

        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids_batch)

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.torch_dtype)

        # Insert image embeddings
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, seq_len] - visual positions only

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    

    @torch.no_grad()
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """Extract understanding features from VLM last layer."""
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, seq_len, vlm_dim]
        return last_layer_features 


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
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    @torch.no_grad()
    def get_answer(
        self,
        messages_list,
        *,
        max_new_tokens=128,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        **generate_kwargs,
    ):
        """
        根据 messages 生成回复文本。支持纯文本或多模态（文本+图像）。

        Args:
            messages_list: 列表，每个元素为一轮对话的 messages（与 chat 模板一致）。
                例如 [{"role": "user", "content": [{"type": "text", "text": "Hello."}]}]
                或含 image 的 [{"type": "image", "image": url_or_pil}, {"type": "text", "text": "Describe this."}]
            max_new_tokens: 最大生成 token 数。
            do_sample: 是否采样；False 为贪心解码。
            temperature: 采样温度（do_sample=True 时有效）。
            top_p: nucleus 采样参数。
            **generate_kwargs: 透传给 model.generate 的其他参数。

        Returns:
            list[str]: 每条 message 对应的生成文本。
        """
        if not messages_list:
            return []
        # 兼容单条：传入一条 message 时包装成 list of list
        if isinstance(messages_list[0], dict) and "role" in messages_list[0]:
            messages_list = [messages_list]

        batch_inputs = []
        for messages in messages_list:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs.append(inputs)

        # 按 batch 拼接或逐条生成（若长度不一则逐条更稳妥）
        input_ids = torch.cat([x["input_ids"] for x in batch_inputs], dim=0)
        attention_mask = torch.cat(
            [
                x.get("attention_mask", torch.ones_like(x["input_ids"]))
                for x in batch_inputs
            ],
            dim=0,
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # pixel_values 等若存在则一并移到 device
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        for k in batch_inputs[0].keys():
            if k in ("input_ids", "attention_mask"):
                continue
            if all(k in x and isinstance(x.get(k), torch.Tensor) for x in batch_inputs):
                model_inputs[k] = torch.cat([x[k] for x in batch_inputs], dim=0).to(
                    self.model.device
                )

        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kw["temperature"] = temperature
            gen_kw["top_p"] = top_p
        gen_kw.update(generate_kwargs)

        generated_ids = self.model.generate(**model_inputs, **gen_kw)

        # 只保留新生成部分
        input_lens = attention_mask.sum(dim=1).long()
        generated_trimmed = [
            out_ids[in_len:].tolist()
            for out_ids, in_len in zip(generated_ids, input_lens)
        ]
        output_texts = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts

    @torch.no_grad()
    def get_vlm_embeddings(
        self,
        text: str,
        images: Optional[ImageSequenceInput] = None,
        *,
        max_images: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """文本 + 可选图像序列 → VLM 最后一层 hidden（定长 pad）。

        Args:
            text: 语言指令。
            images: ``None`` 纯文本；单张 ``PIL.Image`` / 路径；或多帧图像序列（按时间顺序传入）。
            max_images: 最多保留图像数（均匀下采样），防止序列过长 OOM。

        Returns:
            ``(embeddings, attention_mask)``：``embeddings`` 为 ``[1, model_max_length, hidden]``。
        """
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
        """批量：每个样本为 ``(text, images)``。"""
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

    @torch.no_grad()
    def get_text_embeddings(self, texts: str | list[str]):
        """纯文本便捷接口（等价于 ``get_vlm_embeddings(..., images=None)``）。"""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("texts 不能为空")
        samples = [(t, None) for t in texts]
        return self.get_vlm_embeddings_batch(samples)

    @torch.no_grad()
    def get_answer_from_text(self, prompts, **kwargs):
        """
        纯文本问答的便捷接口。每条 prompt 对应一个回复。

        Args:
            prompts: 字符串或字符串列表，用户问题。
            **kwargs: 透传给 get_answer 的参数（如 max_new_tokens, do_sample 等）。

        Returns:
            list[str]: 每个问题对应的生成答案。
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        messages_list = [
            [{"role": "user", "content": [{"type": "text", "text": p}]}]
            for p in prompts
        ]
        return self.get_answer(messages_list, **kwargs)


if __name__ == "__main__":
    model_id = "/mnt/dataset/datasets/cjt_personal/pretrained_models/Qwen3-VL-2B-Instruct/"
    model_max_length = 2048
    debug_png = "/mnt/dataset/projs/projects/RoboTwin/policy/RDT/debug/debug.png"

    encoder = Qwen3Embedder(
        from_pretrained=model_id,
        device="cuda:0",
        model_max_length=model_max_length,
    )

    # 多图单样本：按时间顺序传入帧序列 + 一条指令
    frame_paths = [debug_png, debug_png, debug_png, debug_png]
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    prompt = (
        "These images are consecutive frames from a robot manipulation episode. "
        "Describe what happens across the frames."
    )

    vlm_inputs = encoder._preprocess_vlm_messages(prompt, frames)
    valid_len = int(vlm_inputs["attention_mask"].sum().item())
    hidden_raw = encoder.extract_und_features(vlm_inputs)
    print(f"[multi-image] num_frames={len(frames)} model_max_length={model_max_length}")
    print(f"  raw hidden: {tuple(hidden_raw.shape)}  valid_tokens={valid_len}")

    embeddings, _ = encoder.get_vlm_embeddings(prompt, images=frames)
    print(f"  padded:     {tuple(embeddings.shape)}")
    if valid_len > model_max_length:
        print(f"  truncated:  {valid_len - model_max_length} tokens dropped")
    else:
        print(f"  no truncate ({valid_len} <= {model_max_length})")