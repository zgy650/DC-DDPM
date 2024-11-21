
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
from torch.nn import init

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# 读取.mat文件并计算均值和方差
def load_das_data(mat_file):
    data = scipy.io.loadmat(mat_file)
    das_noise = data['das_noise']
    mean = np.mean(das_noise, axis=(0, 1))  # 计算均值
    std = np.std(das_noise, axis=(0, 1))    # 计算标准差
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32), torch.tensor(das_noise, dtype=torch.float32)

# 使用STFT提取时频特征
def compute_stft(x, n_fft=512, hop_length=256, win_length=512):
    return torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)

# 定义条件高斯扩散训练器
class GaussianDiffusionTrainer_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, mean, std):
        super().__init__()
        self.model = model
        self.T = T
        self.mean = mean
        self.std = std

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, stft_features):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        ct = x_0[:, 0, :, :]  # 取出第一通道
        cbct = x_0[:, 1, :, :]  # 取出第二通道
        ct = torch.unsqueeze(ct, 1)
        cbct = torch.unsqueeze(cbct, 1)

        # 加噪过程
        noise = (ct - self.mean) / self.std  # 使用 DAS 均值和方差归一化噪声
        x_t = (
            extract(self.sqrt_alphas_bar, t, ct.shape) * ct +
            extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise
        )
        x_t = torch.cat((x_t, cbct), 1)

        # 模型预测噪声，使用时频特性作为条件
        predicted_noise = self.model(x_t, t, stft_features)
        loss = F.mse_loss(predicted_noise, noise, reduction='sum')
        return loss

# 定义条件高斯扩散采样器
class GaussianDiffusionSampler_cond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, mean, std):
        super().__init__()
        self.model = model
        self.T = T
        self.mean = mean
        self.std = std

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, stft_features):
        eps = self.model(x_t, t, stft_features)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
        var = extract(self.posterior_var, t, x_t.shape)
        return xt_prev_mean, var

    def forward(self, x_T, stft_features):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t, t, stft_features)

            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise

        x_0 = x_t[:, 0, :, :] * self.std + self.mean  # 反归一化
        return x_0
