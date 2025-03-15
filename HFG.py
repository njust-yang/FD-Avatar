import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# 设置数据路径
data_dir = "path_to_your_face_details"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),         # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # 输入是 latent_dim 维的噪声向量
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # 当前尺寸: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 当前尺寸: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 当前尺寸: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 输出尺寸: img_channels x 64 x 64
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_maps):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 输入尺寸: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前尺寸: (feature_maps*2) x 32 x 32
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前尺寸: (feature_maps*4) x 16 x 16
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前尺寸: (feature_maps*8) x 8 x 8
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
            # 输出尺寸: 1 x 1 x 1
        )

    def forward(self, x):
        return self.net(x)

  # 超参数
latent_dim = 100  # 噪声向量的维度
img_channels = 3   # 图像的通道数（RGB）
feature_maps = 64  # 特征图的数量
lr = 0.0002       # 学习率
beta1 = 0.5        # Adam 优化器的参数

# 初始化生成器和判别器
generator = Generator(latent_dim, img_channels, feature_maps)
discriminator = Discriminator(img_channels, feature_maps)

# 优化器
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# 损失函数
criterion = nn.BCELoss()

# 训练参数
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到设备
generator.to(device)
discriminator.to(device)

# 训练循环
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)

        # 训练判别器
        optimizer_D.zero_grad()
        # 真实图像的损失
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)
        # 生成假图像
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_images = generator(noise)
        # 假图像的损失
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        # 总判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()
        # 生成器希望假图像被判别为真
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # 打印损失
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    # 保存生成的图像
    with torch.no_grad():
        fake_images = generator(torch.randn(16, latent_dim, 1, 1).to(device)).cpu()
        fake_images = (fake_images + 1) / 2  # 反归一化到 [0, 1]
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(np.transpose(fake_images[i], (1, 2, 0)))
            plt.axis("off")
        plt.show()
