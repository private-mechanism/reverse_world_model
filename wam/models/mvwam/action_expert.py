# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 修改位置编码 
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

import re
from collections import OrderedDict

"""
使用siglip视觉编码器, 只有cross attention
"""


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.
    
    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | tensor): a list of positions to be encoded, size (M,).
    Returns:
        out (tensor): resulting positional embedding, size (M, D).
    """
    import numpy as np
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ActionExpertRotary1DPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int=512,
        theta: float = 10000.0,
        scaling: float = 1.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len

        self.dim = attention_head_dim 
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freq_cos, freq_sin = get_1d_rotary_pos_embed(
            self.dim,
            max_seq_len,
            theta,
            use_real=True,
            repeat_interleave_real=True,
            linear_factor=scaling,
            freqs_dtype=freqs_dtype,
        )

        self.register_buffer("freqs_cos", freq_cos, persistent=False)
        self.register_buffer("freqs_sin", freq_sin, persistent=False)

    def forward(self, ppf: int = 0, num_actions: int = 33) -> torch.Tensor:
        freqs_cos = self.freqs_cos[:num_actions].view(1, num_actions, 1, -1)
        freqs_sin = self.freqs_sin[:num_actions].view(1, num_actions, 1, -1)

        return freqs_cos, freqs_sin


class ActionExpertRotary1DPosEmbed_old(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        max_seq_len: int=512,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.a_dim = h_dim + w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [self.t_dim, self.a_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, ppf: int, num_action: int) -> torch.Tensor:
        split_sizes = [self.t_dim, self.a_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_t = torch.cat([freqs_cos[0][:1].view(1, -1), freqs_cos[0][1:ppf].view(ppf-1, -1).repeat((num_action-1)//(ppf-1), 1)], dim=0)
        freqs_cos_a = freqs_cos[1][:num_action].view(num_action, -1).expand(num_action, -1)

        freqs_sin_t = torch.cat([freqs_sin[0][:1].view(1, -1), freqs_sin[0][1:ppf].view(ppf-1, -1).repeat((num_action-1)//(ppf-1), 1)], dim=0)
        freqs_sin_a = freqs_sin[1][:num_action].view(num_action, -1)#.expand(num_action, -1)

        freqs_cos = torch.cat([freqs_cos_t, freqs_cos_a], dim=-1).reshape(1, num_action, 1, -1)
        freqs_sin = torch.cat([freqs_sin_t, freqs_sin_a], dim=-1).reshape(1, num_action, 1, -1)

        return freqs_cos, freqs_sin



def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim, ))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim

    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim, ))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(pos_embed_dim,
                                                                  torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)

    return c_pos_emb


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        visual_len: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        # if attn.add_k_proj is not None:
        #     # 512 is the context length of the text encoder, hardcoded for now
        #     image_context_length = encoder_hidden_states.shape[1] - 512
        #     encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
        #     encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)) # [b, n, h, d]
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query[:, :visual_len] = apply_rotary_emb(query[:, :visual_len], *rotary_emb)
            key[:, :visual_len] = apply_rotary_emb(key[:, :visual_len], *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        # self.add_k_proj = self.add_v_proj = None
        # if added_kv_proj_dim is not None:
        #     self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        #     self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
        #     self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        visual_len: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, visual_len, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        # self.image_embedder = None
        # if image_embed_dim is not None:
        #     self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(self.time_proj.weight.device, self.time_proj.weight.dtype) # .type_as(encoder_hidden_states) # TODO: check if this is correct
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        # if encoder_hidden_states_image is not None:
        #     encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, #encoder_hidden_states_image


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        is_cross_attn: bool = False,
    ):
        super().__init__()

        # 1. Self-attention
        # if not is_cross_attn:
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        # if is_cross_attn:
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=attention_head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=attention_head_dim,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def get_modulation(self, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
            return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
            return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def forward_cross_attn(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
    ):
        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output
        return hidden_states

    def forward_ffn(
        self, 
        hidden_states: torch.Tensor,
        c_scale_msa: torch.Tensor,
        c_shift_msa: torch.Tensor,
        c_gate_msa: torch.Tensor,
    ):
        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states

    def modulate(
        self, 
        hidden_states, 
        residual_hidden_states=None,
        scale=None, 
        shift=None, 
        gate=None
    ):
        if scale is not None and shift is not None:
            return (hidden_states.float() * (1 + scale) + shift).type_as(hidden_states)
        elif gate is not None and residual_hidden_states is not None:
            return (hidden_states.float() + residual_hidden_states.float() * gate).type_as(hidden_states)
        else:
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        visual_len: Optional[int] = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        if hasattr(self, 'attn1'):
            norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb, visual_len)
            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        if hasattr(self, 'attn2'):
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
            hidden_states = hidden_states + attn_output
            # norm_hidden_states = (self.norm2(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
            # attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
            # hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 3. Feed-forward
        if hasattr(self, 'ffn'):
            norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
                hidden_states
            )
            ff_output = self.ffn(norm_hidden_states)
            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class ActionExpertModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        # "rope": {
        #     0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        #     1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        # },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        inner_dim: int = 3072,  
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_action_dim: int = 16,
        out_action_dim: int = 16,
        in_visual_dim: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        max_seq_len: int = 32, 
        num_layers: int = 40,
        visual_encoder_type: str = "siglip2",
        # cross_attn_norm: bool = True,
        eps: float = 1e-6,
        # pos_embed_seq_len: Optional[int] = None,
        # added_kv_proj_dim: Optional[int] = None,
        img_pos_embed_config = None,
        rope_max_seq_len: int = 1024,
        img_cond_len: int=4096,
        use_new_rope: bool = False,
        scaling: float = 4.0,
        use_value: bool = False,
    ) -> None:
        super().__init__()

        # inner_dim = num_attention_heads * attention_head_dim
        out_action_dim = out_action_dim or in_action_dim
        self.hidden_size = inner_dim
        # 1. Patch & position embedding
        if visual_encoder_type == "siglip2":
            self.visual_encoder = self.build_mlp(
                'mlp2x_silu',
                in_features=in_visual_dim,
                out_features=inner_dim,
            )
        # self.state_encoder = self.build_mlp(
        #     'mlp3x_silu',
        #     in_features=in_action_dim,
        #     out_features=inner_dim,
        # )
        self.action_encoder = self.build_mlp(
            'mlp3x_silu',
            in_features=in_action_dim,
            out_features=inner_dim,
        )
        self.action_decoder = self.build_mlp(
            'mlp2x_silu', 
            in_features=inner_dim,
            out_features=out_action_dim,
        )
        if use_value:
            self.value_encoder = self.build_mlp(
                'mlp3x_silu',
                in_features=in_action_dim,
                out_features=inner_dim,
            )
            self.value_decoder = self.build_mlp(
                'mlp2x_silu',
                in_features=inner_dim,
                out_features=out_action_dim,
            )
        # prepare position embeddings for actions
        max_seq_len = max_seq_len
        if use_new_rope:
            if scaling == 1.0:
                self.pos_embedding_rope = ActionExpertRotary1DPosEmbed_old(
                    attention_head_dim=attention_head_dim,
                    max_seq_len=max_seq_len,
                )
            else:
                self.pos_embedding_rope = ActionExpertRotary1DPosEmbed(
                    attention_head_dim=attention_head_dim,
                    max_seq_len=max_seq_len,
                    scaling=scaling,
                )
        else:
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                inner_dim,
                np.arange(max_seq_len)
            )
            pos_embed = torch.from_numpy(pos_embed).float()
            self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

        # prepare position embeddings for image conditions
        self.img_pos_embed_config = img_pos_embed_config
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, torch.arange(img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(embed_dim=self.hidden_size,
                                                               mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                                                               embed_modality=False)
        img_cond_pos_embed = torch.from_numpy(img_cond_pos_embed).float()
        self.register_buffer('img_cond_pos_embed', img_cond_pos_embed.unsqueeze(0))

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            # image_embed_dim=None,
            # pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, attention_head_dim, qk_norm=None, eps=eps, is_cross_attn=False
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        # self.proj_out = nn.Linear(inner_dim, inner_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize linear layers with Xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize output layer
        # nn.init.zeros_(self.decoder.action_head[-1].weight)
        # nn.init.zeros_(self.decoder.action_head[-1].bias)
        
        # Initialize time embedding layers
        nn.init.normal_(self.condition_embedder.time_proj.weight, std=0.02)
        second_linear = self.action_decoder[2]
        if isinstance(second_linear, nn.Linear):
            # 将weight全部置0
            nn.init.zeros_(second_linear.weight)
            # 将bias全部置0（如果有bias）
            if second_linear.bias is not None:
                nn.init.zeros_(second_linear.bias)

        if hasattr(self, 'value_decoder'):
            second_linear = self.value_decoder[2]
            if isinstance(second_linear, nn.Linear):
                # 将weight全部置0
                nn.init.zeros_(second_linear.weight)
                # 将bias全部置0（如果有bias）
                if second_linear.bias is not None:
                    nn.init.zeros_(second_linear.bias)

    def build_mlp(self, projector_type, in_features, out_features):
        """Build MLP projector for encoders."""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_silu_match = re.match(r'^mlp(\d+)x_silu$', projector_type)
            if mlp_silu_match:
                mlp_depth = int(mlp_silu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.SiLU())
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def modulate(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ):
        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        # hidden_states = self.proj_out(hidden_states)
        if hasattr(self, 'value_decoder'):
            hidden_states_value = hidden_states[:, -1:]
            hidden_states = hidden_states[:, :-1]
            out_value = self.value_decoder(hidden_states_value)
        if hasattr(self, 'action_decoder'):
            hidden_states = self.action_decoder(hidden_states)
        if hasattr(self, 'value_decoder'):
            hidden_states = torch.cat([hidden_states, out_value], dim=1)
        return hidden_states

    @classmethod
    def from_custom_config(
        cls,
        config: None,
        wan_transformer: None, 
        load_weights_from_transformer=False,
        img_pos_embed_config=None,
        img_cond_len=4096,
    ):
        action_expert = cls(
            in_action_dim=config.in_action_dim,
            inner_dim=config.inner_dim,
            out_action_dim=config.out_action_dim,
            in_visual_dim=config.in_visual_dim,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            eps=float(config.eps),
            visual_encoder_type=config.visual_encoder_type,
            img_pos_embed_config=img_pos_embed_config,
            img_cond_len=img_cond_len,
            use_new_rope=config.use_new_rope if hasattr(config, 'use_new_rope') else False,
            scaling=config.scaling if hasattr(config, 'scaling') else 4.0,
            use_value=config.use_value if hasattr(config, 'use_value') else False,
        )
        if load_weights_from_transformer:
            # 获取两个模型的 state_dict
            wan_sd = wan_transformer.state_dict()
            expert_sd = action_expert.state_dict()

            # 筛选出 名称和形状都匹配 的参数
            filtered_sd = {}
            for key, param in wan_sd.items():
                if key in expert_sd and expert_sd[key].shape == param.shape:
                    filtered_sd[key] = param

            # 只加载匹配的参数
            # action_expert.load_state_dict(filtered_sd, strict=False)
            key = action_expert.load_state_dict(filtered_sd, strict=False)
            logger.warning(f"action_expert load from Wan-Transformer. missing_keys: {key[0]}")
        return action_expert

    def encode_action_state(self, hidden_states: torch.Tensor, hidden_states_states: torch.Tensor, value: Optional[torch.Tensor] = None):
        hidden_states = torch.cat([hidden_states_states, hidden_states], dim=1)
        hidden_states = self.action_encoder(hidden_states)
        if value is not None and hasattr(self, 'value_encoder'):
            hidden_states_value = self.value_encoder(value)
            hidden_states = torch.cat([hidden_states, hidden_states_value], dim=1)
        if hasattr(self, 'pos_embedding'):
            hidden_states = hidden_states + self.pos_embedding[:, :hidden_states.shape[1]]
        return hidden_states

    def encode_visual_token(self, hidden_states_visual: torch.Tensor):
        hidden_states_visual = hidden_states_visual[:, -self.img_cond_pos_embed.shape[1]:]
        hidden_states_visual = self.visual_encoder(hidden_states_visual)
        hidden_states_visual = hidden_states_visual + self.img_cond_pos_embed[:, :hidden_states_visual.shape[1]]
        return hidden_states_visual

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_states: torch.Tensor,
        hidden_states_visual: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_tokens, num_channels = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size

        rotary_emb = None
        hidden_states_visual = self.visual_encoder(hidden_states_visual)
        hidden_states_visual = hidden_states_visual[:, -self.img_cond_pos_embed.shape[1]:] + self.img_cond_pos_embed[:, :hidden_states_visual.shape[1]]

        visual_len = hidden_states_visual.shape[1]
        action_chunk_size = hidden_states.shape[1]
        # hidden_states = self.action_encoder(hidden_states)
        # hidden_states_states = self.state_encoder(hidden_states_states) 
        hidden_states = torch.cat([hidden_states_states, hidden_states], dim=1)
        hidden_states = self.action_encoder(hidden_states)
        if value is not None and hasattr(self, 'value_encoder'):
            hidden_states_value = self.value_encoder(value)
            hidden_states = torch.cat([hidden_states, hidden_states_value], dim=1)
        if hasattr(self, 'pos_embedding'):
            hidden_states = hidden_states + self.pos_embedding[:, :hidden_states.shape[1]]
        elif hasattr(self, 'pos_embedding_rope'):
            rotary_emb = self.pos_embedding_rope(num_actions=hidden_states.shape[1])

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            # 在第二个维度前面补visual_len个0
            pad_timestep = timestep.new_zeros(timestep.shape[0], hidden_states.shape[1]-timestep.shape[1])
            timestep = torch.cat([pad_timestep, timestep], dim=1)
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states if idx %2 == 0 else hidden_states_visual, timestep_proj, rotary_emb, visual_len
                )
        else:
            for idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states if idx %2 == 0 else hidden_states_visual, timestep_proj, rotary_emb, visual_len)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        # hidden_states = self.proj_out(hidden_states)
        if hasattr(self, 'value_decoder'):
            hidden_states_value = hidden_states[:, -1:]
            hidden_states = hidden_states[:, :-1]
            out_value = self.value_decoder(hidden_states_value)
        output = self.action_decoder(hidden_states[:, -action_chunk_size:, :])
        if hasattr(self, 'value_decoder'):
            output = torch.cat([output, out_value], dim=1)
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # config_path = '/mnt/dataset/projs/projects/RoboTwin/policy/RDT/configs/base_debug.yaml'
    config_path = '/mnt/dataset/projs/projects/RoboTwin/policy/RDT/configs/base_world_flow_sample_d768_n30_c32_s4_rope.yaml'
    config = OmegaConf.load(config_path).model.action_expert
    config.use_new_rope = True
    device = 'cuda'
    video_expert = None
    config.num_layers = 30
    config.inner_dim = 768
    config.use_value = True
    config.ffn_dim = config.inner_dim * 4
    action_expert = ActionExpertModel.from_custom_config(
        config, video_expert, False, img_cond_len=729*6, 
    )
    # 输出action_expert总的参数量
    print(sum(p.numel() for p in action_expert.parameters())/1e6, 'M')
    # import pdb; pdb.set_trace()
    action_expert.to(device)
    hidden_states = torch.randn(1, 16, 14).to(device)
    timestep = torch.randint(0, 100, (1,17)).to(device)
    encoder_hidden_states = torch.randn(1, 16, 4096).to(device)
    hidden_states_visual = torch.randn(1, 729, 1152).to(device)
    hidden_states_states = torch.randn(1, 1, 14).to(device)
    value = torch.ones(1, 1, 14).to(device)
    output = action_expert(hidden_states,hidden_states_states, hidden_states_visual, timestep, encoder_hidden_states, value=value, return_dict=False)[0]
    print(output.shape)