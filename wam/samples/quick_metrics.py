from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _common_latent_slice(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape == target.shape:
        return pred, target
    common = tuple(min(a, b) for a, b in zip(pred.shape, target.shape))
    slices = tuple(slice(0, n) for n in common)
    return pred[slices], target[slices]


def latent_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _common_latent_slice(pred.float(), target.float())
    return F.l1_loss(pred, target).mean()


def _to_video_tensor(video) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(video)).float()
    if tensor.numel() == 0:
        raise ValueError("video metric input is empty")
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def compute_video_l1_psnr_ssim(pred_video, target_video, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    pred = _to_video_tensor(pred_video)
    target = _to_video_tensor(target_video)
    common = tuple(min(a, b) for a, b in zip(pred.shape, target.shape))
    slices = tuple(slice(0, n) for n in common)
    pred = pred[slices]
    target = target[slices]

    l1 = F.l1_loss(pred, target).mean()
    mse = F.mse_loss(pred, target).mean()
    psnr = 10.0 * torch.log10(torch.tensor(1.0, device=mse.device) / mse.clamp_min(eps))

    mu_x = pred.mean()
    mu_y = target.mean()
    var_x = ((pred - mu_x) ** 2).mean()
    var_y = ((target - mu_y) ** 2).mean()
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return {"video_l1": l1, "video_psnr": psnr, "video_ssim": ssim}
