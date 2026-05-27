import torch
from torch.distributions import Normal, Uniform, Beta


class NoiseTimestepSampler:
    """
    统一的时间步采样器，支持以下分布类型：
    - normal: 移位Logit正态分布（原ShiftedLogitNormalTimestepSampler）
    - uniform: 移位Logit均匀分布（原ShiftedLogitNormalTimestepSampler）
    - beta: Beta分布（原BetaTimestepSampler）
    """

    def __init__(
        self,
        distribution_type: str = "beta",
        # 针对 normal/uniform 分布的参数（原ShiftedLogitNormal）
        std: float = 1.0,
        shift: float = 3.0,
        # 针对 beta 分布的参数（原BetaTimestepSampler）
        alpha: float = 1.0,
        beta: float = 1.5,
        noise_s: float = 0.998,
        # 通用参数
        num_train_timesteps: int = 1000
    ):
        """
        初始化统一的时间步采样器
        
        Args:
            distribution_type: 分布类型，可选 "normal"/"uniform"/"beta"
            std: normal分布的标准差（仅normal类型生效）
            shift: normal/uniform分布的移位系数（仅normal/uniform类型生效）
            alpha: beta分布的α参数（仅beta类型生效）
            beta: beta分布的β参数（仅beta类型生效）
            noise_s: beta分布的噪声系数（仅beta类型生效）
            num_train_timesteps: 训练时间步总数（通用参数，保留兼容性）
        """
        # 校验分布类型
        assert distribution_type in ["normal", "uniform", "beta"], \
            f"不支持的分布类型：{distribution_type}，可选：normal/uniform/beta"
        
        self.distribution_type = distribution_type
        self.num_train_timesteps = num_train_timesteps

        # 初始化对应分布的参数和分布实例
        if distribution_type in ["normal", "uniform"]:
            # 原ShiftedLogitNormal参数
            self.std = std
            self.shift = shift
            # 创建分布实例
            if distribution_type == "normal":
                self.dist = Normal(0.0, 1.0)
            elif distribution_type == "uniform":
                self.dist = Uniform(0.0, 1.0)
        
        elif distribution_type == "beta":
            # 原BetaTimestepSampler参数
            self.alpha = alpha
            self.beta = beta
            self.noise_s = noise_s
            # 创建Beta分布实例（移到指定设备时动态处理，避免初始化时设备不匹配）
            self.dist = Beta(
                torch.tensor(self.alpha),
                torch.tensor(self.beta)
            )

    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        采样时间步，保持与原两个类一致的API接口
        
        Args:
            batch_size: 要采样的时间步数量
            device: 采样结果的设备（CPU/CUDA）
        
        Returns:
            Tensor of shape (batch_size,)：采样的时间步，范围适配对应分布逻辑
        """
        timesteps = self.dist.sample((batch_size,))
        if self.distribution_type in ["normal", "uniform"]:
            # 原ShiftedLogitNormal采样逻辑
            
            if self.distribution_type == "normal":
                timesteps = timesteps * self.std
                timesteps = torch.sigmoid(timesteps)
            
            # 移位变换
            timesteps = (timesteps * self.shift) / (1 + (self.shift - 1) * timesteps)

        elif self.distribution_type == "beta":
            # 原BetaTimestepSampler采样逻辑
            
            # Beta分布的数值变换
            timesteps = (self.noise_s - timesteps) / self.noise_s

        if device is not None:
            timesteps = timesteps.to(device=device)
        # 保证timestep在[0, 1]范围内
        # timesteps = (timesteps + 0.01) / 1.02
        timesteps = torch.clamp(timesteps, 0.005, 0.998)
        return timesteps, torch.round(timesteps * self.num_train_timesteps).long()
    
    def set_timesteps(self, num_inference_timesteps: int):
        self.num_inference_timesteps = num_inference_timesteps
        self.timesteps = torch.linspace(self.num_train_timesteps, 0, num_inference_timesteps + 1)
        self.timesteps = self.timesteps[:-1]
        self.sigmas = self.timesteps / self.num_train_timesteps
        return self.timesteps

    def add_noise(self, original_samples, noise, timestep):
        sigma = timestep / self.num_train_timesteps # (b,)
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        device = sample.device
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0.0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * ((sigma_ - sigma).to(device))
        return prev_sample

# if __name__ == "__main__":
#     sampler = NoiseTimestepSampler(distribution_type="beta", alpha=1.5, beta=1.0, noise_s=0.998, num_train_timesteps=1000)
#     sigmas, timesteps = sampler.sample(batch_size=10, device=torch.device("cuda"))
#     print(sigmas)
#     print(timesteps)
#     timesteps = sampler.set_timesteps(num_inference_timesteps=10)
#     print(timesteps)
#     sample = torch.zeros(1, 2, 3)
#     noise = torch.ones(1, 2, 3)
#     timestep = 4
#     for t in timesteps:
#         t = (torch.ones(1,) * t).long()
#         prev_sample = sampler.step(model_output=noise, timestep=t, sample=sample)
#         sample = prev_sample
#         print(prev_sample)
#     prev_sample = sampler.step(model_output=noise, timestep=timestep, sample=sample)
#     print(prev_sample)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # ===================== 1. 配置参数 =====================
    SAMPLE_NUM = 10000  # 每种分布采样10000个点
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    PLOT_BINS = 50  # 直方图的分箱数
    PLOT_FIGSIZE = (15, 5)  # 绘图尺寸

    # ===================== 2. 初始化采样器并采样 =====================
    # 2.1 Normal分布采样器
    normal_sampler = NoiseTimestepSampler(
        distribution_type="normal",
        std=1.0,
        shift=3.0,
        num_train_timesteps=1000
    )
    _, normal_samples = normal_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    # 2.2 Uniform分布采样器
    uniform_sampler = NoiseTimestepSampler(
        distribution_type="uniform",
        shift=3.0,
        num_train_timesteps=1000
    )
    _, uniform_samples = uniform_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    # 2.3 Beta分布采样器
    beta_sampler = NoiseTimestepSampler(
        distribution_type="beta",
        alpha=1.5,
        beta=1.0,
        noise_s=0.999,
        num_train_timesteps=1000
    )
    _, beta_samples = beta_sampler.sample(batch_size=SAMPLE_NUM, device=DEVICE)

    # ===================== 3. 数据预处理（转numpy） =====================
    # 转到CPU并转换为numpy数组（matplotlib仅支持numpy）
    normal_samples_np = normal_samples.cpu().numpy()
    uniform_samples_np = uniform_samples.cpu().numpy()
    beta_samples_np = beta_samples.cpu().numpy()

    # 打印基本统计信息（可选）
    print("=== 采样结果统计信息 ===")
    print(f"Normal分布 - 均值：{np.mean(normal_samples_np):.4f}，标准差：{np.std(normal_samples_np):.4f}")
    print(f"Uniform分布 - 均值：{np.mean(uniform_samples_np):.4f}，标准差：{np.std(uniform_samples_np):.4f}")
    print(f"Beta分布 - 均值：{np.mean(beta_samples_np):.4f}，标准差：{np.std(beta_samples_np):.4f}")

    # ===================== 4. 绘制概率分布直方图 =====================
    plt.rcParams["font.size"] = 12  # 设置全局字体大小
    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE)

    # 4.1 绘制Normal分布
    axes[0].hist(normal_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#1f77b4", edgecolor="black")
    axes[0].set_title("Shifted Logit Normal Distribution", fontsize=14)
    axes[0].set_xlabel("Timestep Value", fontsize=12)
    axes[0].set_ylabel("Probability Density", fontsize=12)
    axes[0].grid(alpha=0.3)

    # 4.2 绘制Uniform分布
    axes[1].hist(uniform_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#ff7f0e", edgecolor="black")
    axes[1].set_title("Shifted Logit Uniform Distribution", fontsize=14)
    axes[1].set_xlabel("Timestep Value", fontsize=12)
    axes[1].set_ylabel("Probability Density", fontsize=12)
    axes[1].grid(alpha=0.3)

    # 4.3 绘制Beta分布
    axes[2].hist(beta_samples_np, bins=PLOT_BINS, density=True, alpha=0.7, color="#2ca02c", edgecolor="black")
    axes[2].set_title("Beta Distribution", fontsize=14)
    axes[2].set_xlabel("Timestep Value", fontsize=12)
    axes[2].set_ylabel("Probability Density", fontsize=12)
    axes[2].grid(alpha=0.3)

    # 调整子图间距
    plt.tight_layout()
    # 保存图片（可选，建议保存为高清格式）
    plt.savefig("timestep_distributions.png", dpi=300, bbox_inches="tight")
    # 显示图片
    plt.show()