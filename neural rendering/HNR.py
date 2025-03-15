import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.activation(self.conv(x))
        return x


class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        # MaxPooling 层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 注意力 MLP
        self.attention_mlp = AttentionMLP(input_dim=64 * 64, hidden_dim=256, output_dim=64 * 64)

        # 特征提取器
        self.feature_extractor_face = FeatureExtractor(in_channels=3, out_channels=64)  # 假设输入是 3 通道的 RGB 图像
        self.feature_extractor_S = FeatureExtractor(in_channels=1, out_channels=64)     # 假设 S 是单通道图像

        # 上采样模块
        self.upsample1 = UpsampleBlock(in_channels=64 * 3, out_channels=256)  # 64*3 是因为 concat(A, B, P)
        self.upsample2 = UpsampleBlock(in_channels=256, out_channels=128)
        self.upsample3 = UpsampleBlock(in_channels=128, out_channels=64)
        self.upsample4 = UpsampleBlock(in_channels=64, out_channels=32)
        self.upsample5 = UpsampleBlock(in_channels=32, out_channels=16)
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # 输出 3 通道图像

    def forward(self, S, face):
        # S 经过 MaxPooling
        S_pooled = self.maxpool(S)  # 输出尺寸: [batch, 1, 32, 32]

        # S 经过注意力 MLP
        batch_size = S_pooled.size(0)
        S_flattened = S_pooled.view(batch_size, -1)  # 展平为 [batch, 32*32]
        A = self.attention_mlp(S_flattened)  # 输出尺寸: [batch, 64*64]
        A = A.view(batch_size, 64, 8, 8)     # 重塑为 [batch, 64, 8, 8]

        # 人脸图经过特征提取
        B = self.feature_extractor_face(face)  # 输出尺寸: [batch, 64, 64, 64]

        # S 经过特征提取
        P = self.feature_extractor_S(S)  # 输出尺寸: [batch, 64, 64, 64]

        # 拼接 A, B, P
        A_resized = F.interpolate(A, size=(64, 64), mode='bilinear', align_corners=True)  # 调整 A 的尺寸
        combined = torch.cat([A_resized, B, P], dim=1)  # 输出尺寸: [batch, 64*3, 64, 64]

        # 上采样到 512x512
        x = self.upsample1(combined)  # [batch, 256, 128, 128]
        x = self.upsample2(x)         # [batch, 128, 256, 256]
        x = self.upsample3(x)         # [batch, 64, 512, 512]
        x = self.upsample4(x)         # [batch, 32, 1024, 1024]
        x = self.upsample5(x)         # [batch, 16, 2048, 2048]
        x = self.final_conv(x)        # [batch, 3, 2048, 2048]

        # 调整到 512x512
        output = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        return output
