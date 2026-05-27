# coding=utf-8
"""因果 World Policy：video / action 联合注意力，支持 value token 与 video KV 屏蔽。"""

from __future__ import annotations

import torch
import torch.nn as nn
from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_utils import ModelMixin

from .action_expert import ActionExpertModel
from .build_model import build_world_stack, compute_img_cond_len_and_pos_embed_config
from .frame_causal_mask import (
    bool_to_additive_mask,
    build_action_self_causal_bool_mask,
    build_action_to_video_frame_causal_bool_mask,
    build_video_frame_causal_additive_mask,
    merge_attention_bias,
)
from .video_expert import WanTransformer3DModel

# action 联合注意力：随机屏蔽视频段 K 的概率（仅 video_kv_mask_mode="random" 时使用）
VIDEO_KV_MASK_PROB = 0.0
# 是否对 k_video 加 mask：random=按概率随机；on=总是；off=从不
VideoKVMaskMode = Literal["random", "on", "off"]


# ---------------------------------------------------------------------------
# 注意力工具
# ---------------------------------------------------------------------------


def video_kv_mask_per_batch(
    mode: VideoKVMaskMode,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """返回 shape (B,) 的 bool：每个样本是否对 video KV 加屏蔽。

    ``random`` 模式下每个 batch 元素独立以 ``VIDEO_KV_MASK_PROB`` 采样，而非整批共用一个随机数。
    """
    if mode == "off":
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    if mode == "on":
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if mode == "random":
        return torch.rand(batch_size, device=device) < VIDEO_KV_MASK_PROB
    raise ValueError(f"Unknown video_kv_mask_mode={mode!r}, expected 'off', 'on', or 'random'.")


def project_attention_qkv(
    attention_module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
):
    """从 Wan attention 模块得到 Q/K/V（与 video_expert 内逻辑一致）。"""
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attention_module.fused_projections:
        if attention_module.cross_attention_dim_head is None:
            query, key, value = attention_module.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attention_module.to_q(hidden_states)
            key, value = attention_module.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attention_module.to_q(hidden_states)
        key = attention_module.to_k(encoder_hidden_states)
        value = attention_module.to_v(encoder_hidden_states)
    return query, key, value


def apply_rotary_embedding(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """video / action 共用的 RoPE（无 video 时 action 仍需要）。"""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


def build_action_joint_attention_mask(
    *,
    base_attention_mask: Optional[torch.Tensor],
    hidden_states_action: torch.Tensor,
    hidden_states_video: Optional[torch.Tensor],
    batch_size: int,
    video_kv_mask_mode: VideoKVMaskMode,
    value: Optional[torch.Tensor],
    action_expert: ActionExpertModel,
    default_attention_backend: Optional[AttentionBackendName],
    action_frame_causal_self_attn: bool = False,
    action_video_frame_causal_kv: bool = False,
    action_prefix_len: int = 0,
    action_token_len: Optional[int] = None,
    action_value_len: int = 0,
    video_num_frames: Optional[int] = None,
    video_tokens_per_frame: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[AttentionBackendName]]:
    """为 action 联合注意力构造 ``attn_mask_action`` 与 backend（与 ``forward`` 内原逻辑一致）。"""
    attn_mask_action = base_attention_mask
    action_attn_backend: Optional[AttentionBackendName] = default_attention_backend
    if attn_mask_action is not None:
        action_attn_backend = AttentionBackendName.NATIVE

    video_token_len = hidden_states_video.shape[1] if hidden_states_video is not None else 0
    action_query_len = hidden_states_action.shape[1]
    key_value_len = video_token_len + action_query_len
    negative_inf = torch.finfo(hidden_states_action.dtype).min

    num_value_tokens = int(action_value_len)
    if value is not None and hasattr(action_expert, "value_encoder"):
        num_value_tokens = max(num_value_tokens, int(value.shape[1]))
    value_query_start = action_query_len - num_value_tokens if num_value_tokens > 0 else None
    action_token_len = (
        int(action_token_len)
        if action_token_len is not None
        else max(0, action_query_len - int(action_prefix_len) - num_value_tokens)
    )

    video_kv_mask_per_sample = video_kv_mask_per_batch(
        video_kv_mask_mode, batch_size, hidden_states_action.device
    )
    mask_value_kv_for_non_value_queries = value_query_start is not None and value_query_start > 0
    use_action_self_causal = bool(action_frame_causal_self_attn)
    use_action_video_frame_causal = bool(action_video_frame_causal_kv and video_token_len > 0)

    if (
        not video_kv_mask_per_sample.any()
        and not mask_value_kv_for_non_value_queries
        and not use_action_self_causal
        and not use_action_video_frame_causal
    ):
        return attn_mask_action, action_attn_backend

    additive_mask = torch.zeros(
        batch_size,
        action_query_len,
        key_value_len,
        device=hidden_states_action.device,
        dtype=hidden_states_action.dtype,
    )
    if video_kv_mask_per_sample.any():
        masked_batch = video_kv_mask_per_sample
        additive_mask[masked_batch, :, :video_token_len] = negative_inf
        if value_query_start is not None:
            # value token 的 query 不屏蔽 video KV（仍可看 video）
            additive_mask[masked_batch, value_query_start:action_query_len, :video_token_len] = 0
    if mask_value_kv_for_non_value_queries:
        additive_mask[:, :value_query_start, video_token_len + value_query_start : key_value_len] = negative_inf
    if use_action_self_causal:
        action_self_allowed = build_action_self_causal_bool_mask(
            prefix_len=int(action_prefix_len),
            action_len=action_token_len,
            value_len=num_value_tokens,
            device=hidden_states_action.device,
        )
        additive_mask[:, :, video_token_len:key_value_len] += bool_to_additive_mask(
            action_self_allowed, hidden_states_action.dtype
        ).unsqueeze(0)
    if use_action_video_frame_causal:
        if video_num_frames is None or video_tokens_per_frame is None:
            raise ValueError(
                "action_video_frame_causal_kv=true requires video_num_frames and video_tokens_per_frame."
            )
        action_to_video_allowed = build_action_to_video_frame_causal_bool_mask(
            query_len=action_query_len,
            prefix_len=int(action_prefix_len),
            action_len=action_token_len,
            value_len=num_value_tokens,
            num_video_frames=int(video_num_frames),
            tokens_per_video_frame=int(video_tokens_per_frame),
            device=hidden_states_action.device,
        )
        if action_to_video_allowed.shape[1] != video_token_len:
            raise ValueError(
                "action/video frame causal mask shape mismatch: "
                f"mask video tokens={action_to_video_allowed.shape[1]}, actual video tokens={video_token_len}"
            )
        additive_mask[:, :, :video_token_len] += bool_to_additive_mask(
            action_to_video_allowed, hidden_states_action.dtype
        ).unsqueeze(0)

    if attn_mask_action is None:
        attn_mask_action = additive_mask
    elif attn_mask_action.dtype == torch.bool:
        bool_mask = attn_mask_action
        if bool_mask.dim() == 2:
            bool_mask = bool_mask.unsqueeze(0).expand(batch_size, -1, -1)
        float_mask = torch.zeros(
            batch_size,
            action_query_len,
            key_value_len,
            device=hidden_states_action.device,
            dtype=hidden_states_action.dtype,
        )
        float_mask.masked_fill_(~bool_mask, negative_inf)
        attn_mask_action = float_mask + additive_mask
    else:
        attn_mask_action = attn_mask_action + additive_mask

    action_attn_backend = AttentionBackendName.NATIVE
    return attn_mask_action, action_attn_backend


def make_forward_joint_checkpoint_fn(
    world_policy: WorldPolicyModel,
    layer_index: int,
    attention_mask: Optional[torch.Tensor],
    attn_mask_action: Optional[torch.Tensor],
    video_attn_backend: Optional[AttentionBackendName],
    action_attn_backend: Optional[AttentionBackendName],
):
    """梯度检查点用：将 tensor 元组解包后调用 ``forward_joint``。"""

    def checkpoint_forward(*tensor_inputs):
        hidden_states_video = tensor_inputs[0] if len(tensor_inputs) > 0 else None
        hidden_states_action = tensor_inputs[1] if len(tensor_inputs) > 1 else None
        hidden_states_visual = tensor_inputs[2] if len(tensor_inputs) > 2 else None
        encoder_hidden_states_video = tensor_inputs[3] if len(tensor_inputs) > 3 else None
        encoder_hidden_states_action = tensor_inputs[4] if len(tensor_inputs) > 4 else None
        timestep_proj_video = tensor_inputs[5] if len(tensor_inputs) > 5 else None
        timestep_proj_action = tensor_inputs[6] if len(tensor_inputs) > 6 else None
        rotary_emb = tensor_inputs[7] if len(tensor_inputs) > 7 else None
        if layer_index < len(world_policy.action_expert.blocks):
            rotary_emb_action = tensor_inputs[8] if len(tensor_inputs) > 8 else None
        else:
            rotary_emb_action = None
        return world_policy.forward_joint(
            layer_index,
            hidden_states_video,
            hidden_states_action,
            hidden_states_visual,
            encoder_hidden_states_video,
            encoder_hidden_states_action,
            timestep_proj_video,
            timestep_proj_action,
            rotary_emb,
            rotary_emb_action,
            attention_mask=attention_mask,
            attn_mask_action=attn_mask_action,
            video_attn_backend=video_attn_backend,
            action_attn_backend=action_attn_backend,
        )

    return checkpoint_forward


def build_forward_joint_checkpoint_inputs(
    layer_index: int,
    num_action_layers: int,
    hidden_states_video,
    hidden_states_action,
    hidden_states_visual,
    encoder_hidden_states_video,
    encoder_hidden_states_action,
    timestep_proj_video,
    timestep_proj_action,
    rotary_emb,
    rotary_emb_action,
):
    """按层索引组装 ``gradient_checkpointing_func`` 的输入列表。"""
    has_action_layer = layer_index < num_action_layers
    inputs = [hidden_states_video]
    if has_action_layer:
        inputs.extend([hidden_states_action, hidden_states_visual])
    else:
        inputs.extend([None, None])
    inputs.append(encoder_hidden_states_video)
    inputs.append(encoder_hidden_states_action if has_action_layer else None)
    inputs.append(timestep_proj_video)
    inputs.append(timestep_proj_action)
    inputs.append(rotary_emb)
    inputs.append(rotary_emb_action if has_action_layer else None)
    return inputs


# ---------------------------------------------------------------------------
# WorldPolicyModel
# ---------------------------------------------------------------------------


class WorldPolicyModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, transformer: WanTransformer3DModel, action_expert: ActionExpertModel):
        super().__init__()
        self.video_expert = transformer
        self.action_expert = action_expert

        self._attention_backend = None
        self._parallel_config = None
        self.action_video_frame_causal_kv = False

        self.gradient_checkpointing_func = None
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        """覆盖父类方法以正确设置梯度检查点函数。"""
        super()._set_gradient_checkpointing(module, value)
        if value and self.gradient_checkpointing_func is None:
            self.gradient_checkpointing_func = partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=False,
            )

    def forward_joint(
        self,
        layer_idx: int,
        hidden_states_video: Optional[torch.Tensor],
        hidden_states_action: Optional[torch.Tensor],
        hidden_states_visual: Optional[torch.Tensor],
        encoder_hidden_states_video: torch.Tensor,
        encoder_hidden_states_action: torch.Tensor,
        timestep_proj: torch.Tensor,
        timestep_proj_act: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb_action: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_action: Optional[torch.Tensor] = None,
        video_attn_backend: Optional[AttentionBackendName] = None,
        action_attn_backend: Optional[AttentionBackendName] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        query_video = key_video = value_video = None
        query_action = key_action = value_action = None
        video_len = 0

        if hidden_states_video is not None:
            video_block = self.video_expert.blocks[layer_idx]

            modulation_video = video_block.get_modulation(timestep_proj)
            norm_video = video_block.norm1(hidden_states_video)
            norm_video = video_block.modulate(
                norm_video,
                scale=modulation_video[1],
                shift=modulation_video[0],
            )

            query_video, key_video, value_video = project_attention_qkv(
                video_block.attn1, norm_video, None
            )
            query_video = video_block.attn1.norm_q(query_video)
            key_video = video_block.attn1.norm_k(key_video)

            num_heads = video_block.attn1.heads
            query_video = query_video.unflatten(2, (num_heads, -1))
            key_video = key_video.unflatten(2, (num_heads, -1))
            value_video = value_video.unflatten(2, (num_heads, -1))
            if rotary_emb is not None:
                query_video = apply_rotary_embedding(query_video, *rotary_emb)
                key_video = apply_rotary_embedding(key_video, *rotary_emb)
            video_len = query_video.shape[1]

        if hidden_states_action is not None:
            action_block = self.action_expert.blocks[layer_idx]

            modulation_action = action_block.get_modulation(timestep_proj_act)
            norm_action = action_block.norm1(hidden_states_action)
            norm_action = action_block.modulate(
                norm_action,
                scale=modulation_action[1],
                shift=modulation_action[0],
            )
            query_action, key_action, value_action = project_attention_qkv(
                action_block.attn1, norm_action, None
            )
            query_action = action_block.attn1.norm_q(query_action)
            key_action = action_block.attn1.norm_k(key_action)
            num_heads = action_block.attn1.heads
            query_action = query_action.unflatten(2, (num_heads, -1))
            key_action = key_action.unflatten(2, (num_heads, -1))
            value_action = value_action.unflatten(2, (num_heads, -1))
            if rotary_emb_action is not None:
                query_action = apply_rotary_embedding(query_action, *rotary_emb_action)
                key_action = apply_rotary_embedding(key_action, *rotary_emb_action)

        key_list, value_list = [], []
        if key_video is not None and value_video is not None:
            key_list.append(key_video)
            value_list.append(value_video)
        if key_action is not None and value_action is not None:
            key_list.append(key_action)
            value_list.append(value_action)

        if not key_list or not value_list:
            return hidden_states_video, hidden_states_action

        key_concat = torch.cat(key_list, dim=1)
        value_concat = torch.cat(value_list, dim=1)

        if query_video is not None and key_video is not None and value_video is not None:
            effective_video_backend = (
                video_attn_backend
                if video_attn_backend is not None
                else (
                    AttentionBackendName.NATIVE
                    if attention_mask is not None
                    else self._attention_backend
                )
            )
            attn_out_video = dispatch_attention_fn(
                query_video,
                key_video,
                value_video,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=effective_video_backend,
                parallel_config=self._parallel_config,
            )
            attn_out_video = attn_out_video.flatten(2, 3)
            attn_out_video = attn_out_video.type_as(query_video)
        else:
            attn_out_video = None

        attn_out_action = None
        if query_action is not None:
            effective_action_backend = (
                action_attn_backend
                if action_attn_backend is not None
                else (
                    AttentionBackendName.NATIVE
                    if attn_mask_action is not None
                    else self._attention_backend
                )
            )
            action_attention_mask = attn_mask_action
            if action_attention_mask is not None and action_attention_mask.dim() == 3:
                action_attention_mask = action_attention_mask.unsqueeze(1)
            attn_out_action = dispatch_attention_fn(
                query_action,
                key_concat,
                value_concat,
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=effective_action_backend,
                parallel_config=self._parallel_config,
            )
            attn_out_action = attn_out_action.flatten(2, 3)
            attn_out_action = attn_out_action.type_as(key_list[0])

        if hidden_states_video is not None:
            attn_out_video = video_block.attn1.to_out[0](attn_out_video)
            attn_out_video = video_block.attn1.to_out[1](attn_out_video)
            hidden_states_video = video_block.modulate(
                hidden_states_video,
                residual_hidden_states=attn_out_video,
                gate=modulation_video[2],
            )
            hidden_states_video = video_block.forward_cross_attn(
                hidden_states_video, encoder_hidden_states_video
            )
            hidden_states_video = video_block.forward_ffn(
                hidden_states_video,
                c_scale_msa=modulation_video[4],
                c_shift_msa=modulation_video[3],
                c_gate_msa=modulation_video[5],
            )

        if hidden_states_action is not None:
            cross_attn_condition = (
                encoder_hidden_states_action
                if layer_idx % 2 == 0
                else hidden_states_visual
            )
            attn_out_action = action_block.attn1.to_out[0](attn_out_action)
            attn_out_action = action_block.attn1.to_out[1](attn_out_action)
            hidden_states_action = action_block.modulate(
                hidden_states_action,
                residual_hidden_states=attn_out_action,
                gate=modulation_action[2],
            )
            hidden_states_action = action_block.forward_cross_attn(
                hidden_states_action, cross_attn_condition
            )
            hidden_states_action = action_block.forward_ffn(
                hidden_states_action,
                c_scale_msa=modulation_action[4],
                c_shift_msa=modulation_action[3],
                c_gate_msa=modulation_action[5],
            )

        return hidden_states_video, hidden_states_action

    def forward(
        self,
        hidden_states_video: Optional[torch.Tensor] = None,
        hidden_states_action: Optional[torch.Tensor] = None,
        hidden_states_robostate: Optional[torch.Tensor] = None,
        hidden_states_visual: Optional[torch.Tensor] = None,
        timestep_video: Optional[torch.LongTensor] = None,
        timestep_action: Optional[torch.LongTensor] = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        video_kv_mask_mode: VideoKVMaskMode = "random",
        return_dict: bool = True,
    ) -> Union[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
        if hidden_states_visual is None:
            hidden_states_visual = encoder_hidden_states
        if hidden_states_video is None and hidden_states_action is None:
            raise ValueError("Both hidden_states_video and hidden_states_action cannot be None simultaneously")

        # --- video 分支预处理 ---
        rotary_emb = None
        video_attention_mask = attention_mask
        video_attn_backend: Optional[AttentionBackendName] = self._attention_backend
        if hidden_states_video is not None:
            batch_size, num_channels, num_frames, height, width = hidden_states_video.shape
            patch_t, patch_h, patch_w = self.video_expert.config.patch_size
            post_patch_num_frames = num_frames // patch_t
            post_patch_height = height // patch_h
            post_patch_width = width // patch_w

            rotary_emb = self.video_expert.rope(hidden_states_video)
            hidden_states_video = self.video_expert.patch_embedding(hidden_states_video)
            hidden_states_video = hidden_states_video.flatten(2).transpose(1, 2)
            if getattr(self.video_expert, "frame_causal_self_attention", False):
                video_frame_causal_mask = build_video_frame_causal_additive_mask(
                    post_patch_num_frames,
                    post_patch_height * post_patch_width,
                    hidden_states_video.device,
                    hidden_states_video.dtype,
                )
                video_attention_mask = merge_attention_bias(
                    video_attention_mask,
                    video_frame_causal_mask,
                    dtype=hidden_states_video.dtype,
                )
            if video_attention_mask is not None:
                video_attn_backend = AttentionBackendName.NATIVE

            if timestep_video is not None:
                if timestep_video.ndim == 2:
                    timestep_seq_len = timestep_video.shape[1]
                    timestep_video = timestep_video.flatten()
                else:
                    timestep_seq_len = None

                temb_video, timestep_proj_video, encoder_hidden_states_video, encoder_hidden_states_image = (
                    self.video_expert.condition_embedder(
                        timestep_video,
                        encoder_hidden_states,
                        encoder_hidden_states_image,
                        timestep_seq_len=timestep_seq_len,
                    )
                )
                if timestep_seq_len is not None:
                    timestep_proj_video = timestep_proj_video.unflatten(2, (6, -1))
                else:
                    timestep_proj_video = timestep_proj_video.unflatten(1, (6, -1))
            else:
                device = hidden_states_video.device
                dtype = hidden_states_video.dtype
                temb_video = torch.zeros((batch_size, self.video_expert.inner_dim), device=device, dtype=dtype)
                timestep_proj_video = torch.zeros(
                    (batch_size, 6, self.video_expert.inner_dim), device=device, dtype=dtype
                )
        else:
            batch_size = hidden_states_action.shape[0] if hidden_states_action is not None else 1
            rotary_emb = None
            temb_video = None
            timestep_proj_video = None
            post_patch_num_frames = post_patch_height = post_patch_width = 1
            patch_t = patch_h = patch_w = 1
            encoder_hidden_states_video = None

        # --- action 分支预处理 ---
        rotary_emb_action = None
        action_prefix_len = 0
        action_token_len = 0
        action_value_len = 0
        if hidden_states_action is not None and hidden_states_robostate is not None:
            action_prefix_len = hidden_states_robostate.shape[1]
            action_token_len = hidden_states_action.shape[1]
            action_value_len = value.shape[1] if value is not None and hasattr(self.action_expert, "value_encoder") else 0
            hidden_states_action = self.action_expert.encode_action_state(
                hidden_states_action, hidden_states_robostate, value=value
            )
            hidden_states_visual = self.action_expert.encode_visual_token(hidden_states_visual)

            if hasattr(self.action_expert, "pos_embedding_rope"):
                rotary_emb_action = self.action_expert.pos_embedding_rope(
                    num_actions=hidden_states_action.shape[1]
                )

            if timestep_action is not None:
                if timestep_action.ndim == 2:
                    timestep_seq_len_action = timestep_action.shape[1]
                    timestep_action = timestep_action.flatten()
                else:
                    timestep_seq_len_action = None

                temb_act, timestep_proj_act, encoder_hidden_states_action = (
                    self.action_expert.condition_embedder(
                        timestep_action,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep_seq_len=timestep_seq_len_action,
                    )
                )
                if timestep_seq_len_action is not None:
                    timestep_proj_act = timestep_proj_act.unflatten(2, (6, -1))
                else:
                    timestep_proj_act = timestep_proj_act.unflatten(1, (6, -1))
            else:
                device = hidden_states_action.device
                dtype = hidden_states_action.dtype
                temb_act = torch.zeros((batch_size, self.action_expert.inner_dim), device=device, dtype=dtype)
                timestep_proj_act = torch.zeros(
                    (batch_size, 6, self.action_expert.inner_dim), device=device, dtype=dtype
                )
        else:
            temb_act = None
            timestep_proj_act = None
            encoder_hidden_states_action = None

        num_layers = (
            len(self.video_expert.blocks)
            if hidden_states_video is not None
            else len(self.action_expert.blocks)
        )
        num_action_layers = len(self.action_expert.blocks)

        attn_mask_action = attention_mask
        action_attn_backend: Optional[AttentionBackendName] = self._attention_backend
        if hidden_states_action is not None:
            attn_mask_action, action_attn_backend = build_action_joint_attention_mask(
                base_attention_mask=attention_mask,
                hidden_states_action=hidden_states_action,
                hidden_states_video=hidden_states_video,
                batch_size=batch_size,
                video_kv_mask_mode=video_kv_mask_mode,
                value=value,
                action_expert=self.action_expert,
                default_attention_backend=self._attention_backend,
                action_frame_causal_self_attn=getattr(
                    self.action_expert, "frame_causal_self_attention", False
                ),
                action_video_frame_causal_kv=getattr(self, "action_video_frame_causal_kv", False),
                action_prefix_len=action_prefix_len,
                action_token_len=action_token_len,
                action_value_len=action_value_len,
                video_num_frames=post_patch_num_frames if hidden_states_video is not None else None,
                video_tokens_per_frame=(
                    post_patch_height * post_patch_width if hidden_states_video is not None else None
                ),
            )

        # --- 逐层联合前向 ---
        for layer_idx in range(num_layers):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                if self._gradient_checkpointing_func is None:
                    self._set_gradient_checkpointing(self.forward_joint, value=True)

                checkpoint_inputs = build_forward_joint_checkpoint_inputs(
                    layer_idx,
                    num_action_layers,
                    hidden_states_video,
                    hidden_states_action,
                    hidden_states_visual,
                    encoder_hidden_states_video,
                    encoder_hidden_states_action,
                    timestep_proj_video,
                    timestep_proj_act,
                    rotary_emb,
                    rotary_emb_action,
                )
                outputs = self.gradient_checkpointing_func(
                    make_forward_joint_checkpoint_fn(
                        self,
                        layer_idx,
                        video_attention_mask,
                        attn_mask_action,
                        video_attn_backend,
                        action_attn_backend,
                    ),
                    *checkpoint_inputs,
                )

                output_index = 0
                if hidden_states_video is not None:
                    hidden_states_video = outputs[output_index]
                    output_index += 1
                if hidden_states_action is not None and layer_idx < num_action_layers:
                    hidden_states_action = outputs[output_index]
            elif (
                hidden_states_action is not None
                and self.action_expert is not None
                and layer_idx < num_action_layers
            ):
                hidden_states_video, hidden_states_action = self.forward_joint(
                    layer_idx,
                    hidden_states_video,
                    hidden_states_action,
                    hidden_states_visual,
                    encoder_hidden_states_video,
                    encoder_hidden_states_action,
                    timestep_proj_video,
                    timestep_proj_act,
                    rotary_emb,
                    rotary_emb_action,
                    attention_mask=video_attention_mask,
                    attn_mask_action=attn_mask_action,
                    video_attn_backend=video_attn_backend,
                    action_attn_backend=action_attn_backend,
                )
            else:
                hidden_states_video, _ = self.forward_joint(
                    layer_idx,
                    hidden_states_video,
                    None,
                    None,
                    encoder_hidden_states_video,
                    None,
                    timestep_proj_video,
                    None,
                    rotary_emb,
                    rotary_emb_action,
                    attention_mask=video_attention_mask,
                    attn_mask_action=attn_mask_action,
                    video_attn_backend=video_attn_backend,
                    action_attn_backend=action_attn_backend,
                )

        # --- 输出后处理 ---
        if hidden_states_video is not None and temb_video is not None:
            hidden_states_video = self.video_expert.modulate(hidden_states_video, temb_video)
            hidden_states_video = hidden_states_video.reshape(
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                patch_t,
                patch_h,
                patch_w,
                -1,
            )
            hidden_states_video = hidden_states_video.permute(0, 7, 1, 4, 2, 5, 3, 6)
            hidden_states_video = hidden_states_video.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if hidden_states_action is not None and temb_act is not None:
            hidden_states_action = self.action_expert.modulate(hidden_states_action, temb_act)
            hidden_states_action = hidden_states_action[:, hidden_states_robostate.shape[1] :]

        if not return_dict:
            return (hidden_states_video, hidden_states_action)

        return {
            "hidden_states_video": hidden_states_video,
            "hidden_states_action": hidden_states_action,
        }


if __name__ == "__main__":
    # 本地烟测: python -m wam.models.mvwam.world_policy_causal_value
    from pathlib import Path

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "configs" / "base_world_flow_sample_d768_n30_c32_s4_rope--128dim--value.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到默认配置: {config_path}")
    full_config = OmegaConf.load(config_path)
    model_cfg = full_config.model
    common_cfg = full_config.common

    img_cond_len, img_pos_embed_config = compute_img_cond_len_and_pos_embed_config(common_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    world_model = build_world_stack(
        config=model_cfg,
        img_cond_len=img_cond_len,
        img_pos_embed_config=img_pos_embed_config,
        pretrained_video_expert_path=None,
        pretrained_action_expert_path=None,
        pretrained_wam_path=None,
        dtype=weight_dtype,
        pretrained_video_expert_base_path=None,
    )
    world_model = world_model.to(device=device, dtype=weight_dtype)

    action_dim = int(model_cfg.action_expert.in_action_dim)
    state_dim = int(model_cfg.state_token_dim)
    batch_size = 2
    noisy_action = torch.randn(batch_size, 32, state_dim, device=device, dtype=weight_dtype)
    action_mask = torch.randn(batch_size, 32, state_dim, device=device, dtype=weight_dtype)
    hidden_states_action = torch.cat([noisy_action, action_mask], dim=2)
    state_tokens = torch.randn(batch_size, 1, state_dim, device=device, dtype=weight_dtype)
    state_mask = torch.randn(batch_size, 1, state_dim, device=device, dtype=weight_dtype)
    hidden_states_robostate = torch.cat([state_tokens, state_mask], dim=2)
    noisy_value = torch.randn(batch_size, 1, state_dim, device=device, dtype=weight_dtype)
    value = torch.cat([noisy_value, state_mask], dim=2)

    hidden_states_video = torch.randn(batch_size, 36, 33, 30, 30, device=device, dtype=weight_dtype)
    timestep_video = torch.randint(0, 100, (batch_size,), device=device)
    timestep_action = torch.randint(0, 100, (batch_size,), device=device)
    encoder_hidden_states = torch.randn(
        batch_size, 512, int(model_cfg.lang_token_dim), device=device, dtype=weight_dtype
    )
    hidden_states_visual = torch.randn(
        batch_size, img_cond_len, int(model_cfg.img_token_dim), device=device, dtype=weight_dtype
    )

    with torch.no_grad():
        output = world_model(
            hidden_states_video,
            hidden_states_action,
            hidden_states_robostate,
            hidden_states_visual,
            timestep_video,
            timestep_action,
            encoder_hidden_states,
            video_kv_mask_mode="on",
            value=value,
            return_dict=False,
        )
    print("build_world_stack smoke OK:", output[0].shape, output[1].shape)
