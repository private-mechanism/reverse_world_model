# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
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

import torch
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import List, Optional, Any
import math
import ssl
import torchvision
from torch import Tensor
from src.methods.utils import is_main_process
import torch.nn.functional as F


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)



def build_backbone(
    hidden_dim, position_embedding, lr_backbone, masks, backbone, dilation, use_frozen_bn=False
):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation, use_frozen_bn)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_position_encoding(hidden_dim, pos_emb):
    N_steps = hidden_dim // 2
    if pos_emb in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif pos_emb in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {pos_emb}")

    return position_embedding



class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than
    torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias




class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later
        # layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in
        # name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class TrainableBatchNorm2d(nn.BatchNorm2d):
    """
    Trainable BatchNorm2d that allows gradient flow and parameter updates.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        # 确保在训练模式下更新running statistics
        if self.training:
            return super().forward(x)
        else:
            # 在推理模式下使用训练好的running statistics
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                False, 0, self.eps
            )


class Backbone(BackboneBase):
    """ResNet backbone with trainable BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        use_frozen_bn: bool = False,
    ):
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # 根据参数选择BatchNorm类型
        if use_frozen_bn:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = TrainableBatchNorm2d
        
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, task_emb: Optional[Any] = None):
        if task_emb is not None:
            xs = self[0](tensor_list, task_emb=task_emb)
            # Make a dictionary out of the last layer outputs
            # since we don't have IntermediateLayerGetter
            xs = {"0": xs}
        else:
            xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos
