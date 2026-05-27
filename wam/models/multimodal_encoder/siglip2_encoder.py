import torch

import torch.nn as nn
from transformers import AutoConfig, SiglipImageProcessor, SiglipVisionModel
from diffusers.utils import load_image

class SiglipVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = (
            getattr(args, "mm_vision_select_feature", "patch") if args is not None else "patch"
        )

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        if device_map is not None:
            self.vision_tower = SiglipVisionModel.from_pretrained(
                self.vision_tower_name, device_map=device_map
            )
        else:
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.eval()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs.last_hidden_state  # (B, 729, 1536)
        elif self.select_feature == 'cls_patch':
            image_features = image_forward_outs.pooler_output  # (B, 1, 1536)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2

if __name__ == "__main__":
    encoder = SiglipVisionTower(
        vision_tower="/mnt/dataset/ckpt/pretrained_models/siglip2-so400m-patch14-384",
        args=None,
        delay_load=False
    )
    image = load_image("/mnt/dataset/projs/projects/Wan-Trainer/examples/demo.jpg")
    image = encoder.image_processor(image, return_tensors="pt")['pixel_values']
    image_features = encoder(image)
    # print(image_features)
    print(f" image.shape          = ", image.shape)          # torch.Size([1, 3, 384, 384])
    print(f" image_features.shape = ", image_features.shape) # torch.Size([1, 729, 1152])