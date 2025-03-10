# %%
import os
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.io import savemat
import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

from datasets import *
import numpy as np
import nibabel as nib
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# %%
wandb.login()

# %%
# 自定义数据集类，用于加载和处理医学图像数据
class ImageDatasets(Dataset):
    def __init__(self, path):
        # 初始化时加载指定路径下的所有图像文件
        # os.listdir(path) 返回指定路径下所有文件的列表
        # 然后使用 os.path.join 生成每个图像文件的完整路径
        self.images = [os.path.join(path, img) for img in os.listdir(path)]

        # 打印信息，确认数据集加载完成
        print('"End reading dataset..."')

    # 该方法用于将每张图像切分成左右两部分
    def apple_LR(self, img):
        img = img[np.newaxis, :]  # 在第一个维度新增一个维度，使得它变为四维数组

        # 将图像按硬编码的坐标切分成左图(l_img)和右图(r_img)
        # 切割的范围基于图像的维度：深度（depth）、高度（height）和宽度（width）
        l_img = img[:, 5:60, 15:134, 15:102]  # 左图：深度为5到60，高度和宽度固定
        r_img = img[:, 60:115, 15:134, 15:102]  # 右图：深度为60到115，高度和宽度固定

        return l_img, r_img, img  # 返回左图、右图和原始图像

    # __getitem__ 是 Dataset 类必须实现的方法，用于根据索引返回数据集中的一个样本
    def __getitem__(self, index):
        # 使用 nibabel 加载图像文件，nib.load 返回一个 NIfTI 文件对象
        img = nib.load(self.images[index])

        # 获取图像的数据部分，并使用 np.squeeze 移除多余的单维度
        img = np.squeeze(img.get_data())  # 使用 get_data() 获取图像的原始数据

        # 使用一个阈值（0.05）将图像中的低强度像素去除，只保留大于0.05的像素值
        pos = img > 0.05  # 生成一个布尔数组，只有强度大于0.05的位置为True
        img = img * pos  # 将低于0.05的像素置零

        # 将图像分割成左右两部分
        l_img, r_img, raw_img = self.apple_LR(img)

        # 返回三个图像部分：左图、右图和原始图像，均转换为 PyTorch 的 tensor 类型
        # 使用 torch.from_numpy 将 numpy 数组转换为 PyTorch 的张量
        return torch.from_numpy(l_img), torch.from_numpy(r_img), torch.from_numpy(raw_img)

    # __len__ 是 Dataset 类必须实现的方法，用于返回数据集的大小（即样本数量）
    def __len__(self):
        # 返回图像列表的长度，即数据集中的样本数量
        return len(self.images)
    def save_nifti(self, img, prefix):
        """将图像保存为 NIfTI 文件"""
        # 转换为 NumPy 数组
        img_np = np.array(img.squeeze())  # 去除多余的维度
        # 创建仿射矩阵（假设没有特别要求）
        AFFINE = np.array([[-1.5, 0., 0., 90.],
                            [0., 1.5, 0., -126.],
                            [0., 0., 1.5, -72.],
                            [0., 0., 0., 1.]])
        
        # 保存为 NIfTI 文件
        img_nifti = nib.Nifti1Image(img_np, affine=AFFINE)
        nifti_path = f"./data/cut/{prefix}.nii.gz"
        nib.save(img_nifti, nifti_path)
        print(f"{prefix} image saved to {nifti_path}")
###

# %%
# # 假设你已经定义了 ImageDatasets 类

# # 使用自定义数据集类加载和处理图像
# dataset_path = "./data/train"  # 请替换为你的实际路径
# dataset = ImageDatasets(dataset_path)

# # 获取数据集中的一个样本并切割
# l_img, r_img, raw_img = dataset[2]  # 获取第一个图像

# # 这里只是验证图像的形状
# print(f"Left image shape: {l_img.shape}")
# print(f"Right image shape: {r_img.shape}")
# print(f"Raw image shape: {raw_img.shape}")

# # 保存切割后的图像为 NIfTI 文件（你可以根据需要修改路径）
# dataset.save_nifti(l_img, "LH_GM")
# dataset.save_nifti(r_img, "RH_GM")
# dataset.save_nifti(raw_img, "GM")



# %%
class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用3D卷积进行下采样
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 3D卷积，步幅2，填充1
            nn.BatchNorm3d(out_channels),  # 3D批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    def forward(self, x):
        return self.conv(x)  # 前向传播，执行下采样操作
    

# 3D上采样模块
class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels, stri, pad, out_pad):
        super().__init__()
        # 使用3D转置卷积进行上采样
        self.dcov = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stri, padding=pad,
                               output_padding=out_pad),  # 3D转置卷积
            nn.BatchNorm3d(out_channels),  # 3D批归一化
            nn.LeakyReLU(inplace=True)  # LeakyReLU激活函数
        )

    def forward(self, x):
        return self.dcov(x)  # 前向传播，执行3D上采样

# 保存网络参数
def save_parm(param1, save_name):
    param1 = param1.cpu()  # 将参数移到CPU
    param1 = param1.numpy()  # 转换为NumPy数组
    # 可以保存更多参数，这里只保存param1
    savemat(save_name, {'FM': param1})  # 将参数保存为.mat文件


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 下采样阶段
        self.down1 = Down3d(1, 64)  # 输入1通道，输出64通道
        self.down2 = Down3d(64, 64)  # 输入64通道，输出64通道
        self.down3 = Down3d(64, 128)  # 输入64通道，输出128通道
        self.down4 = Down3d(128, 256)  # 输入128通道，输出256通道
        self.bottle = nn.Conv3d(256, 5000, 1)  # 瓶颈层，1x1x1卷积（不改变空间尺寸）

        # 上采样阶段
        self.up1 = Up3d(5000, 256, 2, 1, 0)  # 3D转置卷积，上采样
        self.up2 = Up3d(256, 128, 2, 1, 1)
        self.up3 = Up3d(128, 64, 2, 1, 1)
        self.up4 = Up3d(64, 64, 2, 1, 1)

        # 输出层
        self.out = nn.Sequential(
            nn.Conv3d(64, 1, 2, 1, 0),  # 输出层，生成最终图像
            nn.Sigmoid()  # 使用Sigmoid激活函数输出值范围[0, 1]
        )

    def forward(self, inp, train=True, save_name=''):
        if train:
            # 训练模式
            x = self.down1(inp)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.bottle(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.out(x)  # 输出生成的图像
        else:
            # 测试模式
            x = self.down1(inp)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.bottle(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.out(x)
            # 在测试时保存参数（可选）
            # save_parm(x.detach(), save_name)
        return x  # 返回生成的图像

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 3D卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, bias=False),  # 输入1通道，输出32通道
            nn.BatchNorm3d(32),  # 批归一化
            nn.LeakyReLU(inplace=True)  # LeakyReLU激活函数
        )

        # 3D卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True)
        )

        # 3D卷积块3
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )

        # 3D卷积块4
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )

        # 全连接层，输出一个概率值，表示图像真假
        self.fc = nn.Linear(128 * 2 * 6 * 4, 1)

    def forward(self, img):
        # 前向传播，依次通过卷积块进行处理
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)  # 将输出展平为一维
        x = self.fc(x)  # 通过全连接层输出概率
        return x  # 返回真假判定结果


# %%
# 设置训练参数（可以直接在此设置）
n_epochs = 300  # 训练的epoch数量
batch_size = 32  # 每个batch的大小
lr = 0.001  # 学习率
g_lr = 0.002
d_lr = 0.0008
n_critic = 1
n_cpu = 8  # CPU的线程数
latent_dim = 100  # 潜在空间的维度
sample_interval = 140  # 生成样本的间隔
out_path = "./data/recon/"  # 生成结果保存的路径
semi = "L"  # 半监督模式，'L'表示左图
g_adv_p = 0.002 # 生成器对抗损失的权重
g_voxel_p = 0.998

# %%
wandb.init(
    project = "Asy-GANres-L",
    config = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "g_lr": g_lr,
        "d_lr": d_lr,
        "n_critic":n_critic,
        "g_adv_p": g_adv_p,
        "g_voxel_p": g_voxel_p,
        "n_cpu": n_cpu,
        "latent_dim": latent_dim,
        "sample_interval": sample_interval,
        "out_path": out_path,
        "semi": semi,

        
    
    }
)

# %%
# 检查是否使用GPU
cuda = True if torch.cuda.is_available() else False
lambda_gp = 10  # WGAN-GP中的梯度惩罚系数


# %%
cuda

# %%
# 权重初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)  # 使用He初始化卷积层权重
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # 使用正态分布初始化BatchNorm层
        torch.nn.init.constant_(m.bias.data, 0.0)  # 将偏置初始化为0

# %%
# 损失函数：对抗损失（用于判别器和生成器）和体素级损失（用于重建）
adversarial_loss = torch.nn.BCELoss()
voxelwise_loss = torch.nn.L1Loss()

# %%
# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()
print("# parameters:", sum(param.numel() for param in generator.parameters()))  # 打印生成器的总参数量
print("# parameters:", sum(param.numel() for param in discriminator.parameters()))

# %%
# 如果有多个GPU，使用DataParallel进行并行训练
if torch.cuda.device_count() > 1:
    print("let's use ", torch.cuda.device_count(), "GPUs")
    generator = torch.nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
    discriminator = torch.nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))

# %%
# 将所有模型、损失函数移到GPU上
if cuda:
    adversarial_loss.cuda()
    voxelwise_loss.cuda()
    generator.cuda()
    discriminator.cuda()

# %%
# 初始化生成器和判别器的权重
generator.apply(weights_init_normal)


# %%
discriminator.apply(weights_init_normal)

# %%

# generator.load_state_dict(torch.load("./generator_epoch_500_1st.pth")['state_dict'])
# print("continue train netG")


# discriminator.load_state_dict(torch.load("./discriminator_epoch_500_1st.pth")['state_dict'])
# print("continue train netD")


# %%
# 加载训练数据和测试数据
data_train = DataLoader(ImageDatasets("./data/train/"),
                        batch_size=batch_size,
                        shuffle=True, num_workers=n_cpu, drop_last=True)
data_test = DataLoader(ImageDatasets("./data/val/"), batch_size=70,
                       shuffle=False, num_workers=1)

# %%
# 定义优化器（生成器和判别器都使用Adam优化器）
optimizer_G = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

# 定义学习率调度器
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=0.0001)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=n_epochs, eta_min=0.0001)


# %%
# 根据是否使用GPU来选择Tensor类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# %%
# 用于保存NIfTI图像的仿射矩阵
AFFINE = np.array([[-1.5, 0., 0., 90.],
                   [0., 1.5, 0., -126.],
                   [0., 0., 1.5, -72.],
                   [0., 0., 0., 1.]])

# %%
# 计算WGAN-GP中的梯度惩罚
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake1 = Tensor(real_samples.shape[0], 1).fill_(1.0)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake1,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# %%
# 保存生成的样本图像
def save_sample(ite):
    generator.eval()
    with torch.no_grad():
        l_img, r_img, raw_img = next(iter(data_test))
        l_img = l_img.type(Tensor)
        r_img = r_img.type(Tensor)
        generator.eval()
        if semi == 'L':
            gen_mask = generator(l_img)
            err = voxelwise_loss(gen_mask, r_img)
        else:
            gen_mask = generator(r_img)
            err = voxelwise_loss(gen_mask, l_img)
    print('test_MSEloss:', err.item())
    # 将生成的图像嵌入原始图像
    if semi == 'L':
        raw_img[:, :, 60:115, 15:134, 15:102] = gen_mask
    else:
        raw_img[:, :, 5:60, 15:134, 15:102] = gen_mask
    raw_img = raw_img.cpu()
    raw_img = raw_img.detach()
    for index in np.arange(3):
        recon = raw_img[index, :, :, :, :].numpy()
        recon = np.squeeze(recon)
        img = nib.Nifti1Image(recon, affine=AFFINE)
        nib.save(img, "./data/recon/%d_%d.nii" % (ite, index))

# %%
# 设置判别器的更新次数
  # 每训练一次生成器，训练判别器五次

# 开始训练
for epoch in range(n_epochs):
    for i, (l_img, r_img, raw_img) in enumerate(data_train):
        generator.train()
        discriminator.train()

        l_img = l_img.type(Tensor)
        r_img = r_img.type(Tensor)
        if semi == 'L':
            masked_data = r_img  # 被遮挡的图像数据
            masked_part = l_img  # 用来生成的部分（左图）
        else:
            masked_data = l_img  # 被遮挡的图像数据
            masked_part = r_img  # 用来生成的部分（右图）

        # 训练生成器
        optimizer_G.zero_grad()
        gen_part = generator(masked_part)

        g_adv = -torch.mean(discriminator(gen_part))  # 生成器的对抗损失
        g_voxel = voxelwise_loss(gen_part, masked_data)  # 生成器的体素损失
        g_loss = g_adv_p * g_adv + g_voxel_p * g_voxel  # 总生成器损失

        g_loss.backward()
        optimizer_G.step()
        # 训练判别器（每次训练生成器后训练判别器五次）
        for _ in range(n_critic):  # 每训练一次生成器，训练判别器五次
        # 训练判别器
            optimizer_D.zero_grad()
            gradient_penalty = compute_gradient_penalty(discriminator, masked_data.data, gen_part.data)
            d_loss = -torch.mean(discriminator(masked_data)) + torch.mean(
                discriminator(gen_part.detach())) + lambda_gp * gradient_penalty  # 判别器的损失
            d_loss.backward()
            optimizer_D.step()
        wandb.log(
            {
                "d_loss": d_loss.item(), 
                "g_loss": g_loss.item(), 
                "g_adv_loss": g_adv.item(), 
                "g_voxel_loss": g_voxel.item(),
                "gradient_penalty": gradient_penalty.item(),
                "g_lr": optimizer_G.param_groups[0]['lr'],
                "d_lr": optimizer_D.param_groups[0]['lr'],
            }
        )        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [GP: %f] [G adv: %f, pixel: %f]"
            % (epoch + 1, n_epochs, i + 1, len(data_train), d_loss.item(), gradient_penalty.item(), g_adv.item(),
               g_voxel.item())
        )

        batches_done = epoch * len(data_train) + i + 1
        if batches_done % sample_interval == 0:
            save_sample(batches_done)  # 每隔一段时间保存样本
    
    # 更新学习率
    scheduler_G.step()
    scheduler_D.step()        

    if (epoch + 1) % 100 == 0:
        # 构造保存文件的路径
        out_pathG = f'./generator_epoch_{epoch + 1}.pth'  # 生成器模型文件名，包含epoch编号
        out_pathD = f'./discriminator_epoch_{epoch + 1}.pth'  # 判别器模型文件名，包含epoch编号
        # 保存生成器模型
        torch.save({
            'epoch': epoch + 1,
            'state_dict': generator.state_dict()
        }, out_pathG)

        # 保存判别器模型
        torch.save({
            'epoch': epoch + 1,
            'state_dict': discriminator.state_dict()
        }, out_pathD)

        print(f"Models saved at epoch {epoch + 1} to {out_pathG} and {out_pathD}")

# %%
wandb.finish()






