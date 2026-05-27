import torch
from diffusers import AutoencoderKLWan

import os, sys
import numpy as np
from pathlib import Path
# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent))
os.environ["TRANSFORMERS_ALLOW_TORCH_LOAD_WITH_UNSAFE_WEIGHTS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

class VAEEncoder:
    def __init__(self, model_path):
        self.model = AutoencoderKLWan.from_pretrained(
            model_path, 
            subfolder="vae", 
        )
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode_to_latents(self, video, vae_mini_batch=1):
        """
        VAE的时间压缩规则：
        - 第1次编码: 原始帧[0] → latent[0]
        - 第2次编码: 原始帧[1:5] → latent[1]
        - 第3次编码: 原始帧[5:9] → latent[2]
        - ...
        - 总压缩比: 1 + (F-1)//4，例如140帧 → 35个latent时间步)(i.e, 不足 4 帧的尾巴 → 会直接被丢掉)
        """
        # 把 video 转到和 model 一样的 device 和 dtype 上 (通常是 GPU 和 bfloat16)，以加速编码过程
        video = video.to(self.model.device, self.model.dtype) # [1, 3, 140, 240, 320]
        def _slice_vae(pixel_values):
            bs = vae_mini_batch
            new_pixel_values = []
            for i in range(0, pixel_values.shape[0], bs):
                pixel_values_bs = pixel_values[i : i + bs]
                pixel_values_bs = self.model.encode(pixel_values_bs).latent_dist
                pixel_values_bs = pixel_values_bs.sample()
                new_pixel_values.append(pixel_values_bs)
            return torch.cat(new_pixel_values, dim = 0)
        latents = _slice_vae(video)
        latents_mean = (
            torch.tensor(self.model.config.latents_mean)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(1, self.model.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = (latents - latents_mean) * latents_std
        return latents # [1, 16, 35, 30, 40]

    @torch.no_grad()
    def decode_to_video(self, latents, vae_mini_batch=1, to_save=False):
        # 把 latents 转到和 model 一样的 device 和 dtype 上 (通常是 GPU 和 bfloat16)，以加速解码过程
        latents = latents.to(self.model.device, self.model.dtype) # [1, 16, 35, 30, 40]
        def _slice_vae_decode(latents):
            bs = vae_mini_batch
            decoded_frames = []
            for i in range(0, latents.shape[0], bs):
                latents_bs = latents[i : i + bs]
                latents_mean = (
                    torch.tensor(self.model.config.latents_mean)
                    .view(1, self.model.config.z_dim, 1, 1, 1)
                    .to(latents_bs.device, latents_bs.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(
                    1, self.model.config.z_dim, 1, 1, 1
                ).to(latents_bs.device, latents_bs.dtype)
                
                
                latents_bs = latents_bs / latents_std + latents_mean
                
                decoded = self.model.decode(latents_bs).sample
                decoded_frames.append(decoded)
            return torch.cat(decoded_frames, dim=0)
        video = _slice_vae_decode(latents) # [1, 3, 137, 240, 320]

        if to_save:
            video_list = []
            for sub_video in video:
                video_np = (sub_video.permute(1,2,3,0).cpu().float() * 127.5 + 127.5).clamp(0,255).numpy().astype(np.uint8)
                video_list.append(video_np)
            return video_list # (137, 240, 320, 3)

        return video
    
    @torch.no_grad()
    def get_condition(self, video, video_latents=None, vae_mini_batch=1):
        """
        为视频生成任务创建条件信息

        输入：完整视频(video) [B, 3, F, H, W]
        输出：条件张量(condition) [B, scale_factor_temporal+z_dim, T_lat, H_lat, W_lat]

        若已算 ``video_latents``，仅对首帧做 VAE（避免对「首帧+全零帧」再跑一遍完整 encode，训练可省约一半 VAE 时间）。
        """
        video = video.to(self.model.device, self.model.dtype)
        B, _, F, H, W = video.shape
        first_frame = video[:, :, :1]

        if video_latents is not None:
            T_lat = video_latents.shape[2]
            z_dim = video_latents.shape[1]
            first_lat = self.encode_to_latents(first_frame, vae_mini_batch=vae_mini_batch)
            H_lat, W_lat = first_lat.shape[-2], first_lat.shape[-1]
            condition_video_latents = first_lat.new_zeros(B, z_dim, T_lat, H_lat, W_lat)
            n_copy = min(first_lat.shape[2], T_lat)
            condition_video_latents[:, :, :n_copy] = first_lat[:, :, :n_copy]
        else:
            condition_video = torch.cat(
                [first_frame, first_frame.new_zeros(B, 3, F - 1, H, W)],
                dim=2,
            )
            condition_video_latents = self.encode_to_latents(
                condition_video, vae_mini_batch=vae_mini_batch
            )
        T_lat = condition_video_latents.shape[2]
        H_lat = condition_video_latents.shape[-2]
        W_lat = condition_video_latents.shape[-1]

        # 构造 mask (scale_factor_temporal=4 个通道，标记第0个latent时间步为条件)
        # 等价于 ori 版 repeat_interleave→view→transpose 后的结果:
        #   第0个时间步所有4个通道=1 (条件), 其余时间步所有4个通道=0 (需要生成)
        mask_lat_size = torch.zeros(B, self.model.config.scale_factor_temporal, T_lat, H_lat, W_lat,
                                    device=video.device, dtype=video.dtype) # [B, 4, T_lat, H_lat, W_lat] = [1, 4, 9, 30, 40]
        mask_lat_size[:, :, 0:1, :, :] = 1.0

        # 4: 拼接mask和latents (在通道维度)
        condition = torch.cat([
            mask_lat_size,          # [B, 4,  T_lat, H_lat, W_lat] mask通道 ([1, 4, 9, 30, 40])
            condition_video_latents # [B, 16, T_lat, H_lat, W_lat] latent通道 ([1, 16, 9, 30, 40])
        ], dim=1) # [B, 20, T_lat, H_lat, W_lat] = [1, 20, 9, 30, 40] 或 [1, 20, 35, 30, 40]
        return condition

if __name__ == "__main__":
    import numpy as np
    import imageio
    from diffusers.utils import load_video, export_to_video
    model_path = "/mnt/dataset/projs/pretrained_models/WoW-1-Wan-1.3B-2M-Diffusers"
    video_path = "/mnt/dataset/projs/projects/RoboTwin/data/adjust_bottle/aloha-agilex_clean_50/video/episode0.mp4"
    vae_encoder = VAEEncoder(model_path)
    # 把 model 转到 GPU 上，并使用 bfloat16 精度来加速计算
    vae_encoder.model = vae_encoder.model.to("cuda", dtype=torch.bfloat16)
    video = load_video(video_path)
    video = np.stack([np.array(frame) for frame in video], axis=0)   # (140, 240, 320, 3)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0) # [1, 3, 140, 240, 320]
    
    # (3) 测试 decode 出来的视频质量
    # video_np = vae_encoder.decode_to_video(latents, vae_mini_batch=256, to_save=True) # (137, 240, 320, 3)
    # imageio.mimwrite("output_video.mp4", video_np, fps=30, codec='libx264')
    # print(f"Decoded video saved to output_video.mp4, shape = {video_np.shape}")