from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# load_fer2013为进行了10折裁剪处理的版本，load_fer2013_bk为未进行10折裁剪处理的版本
def load_fer2013(batch_size=64, image_size=48, root="/home/ivan/XDU/code/Facial-Expression-Recognition.Pytorch-master/data"):
    """
    严格匹配原仓库数据处理流程，修复准确率低的问题
    """
    class FER2013Dataset(Dataset):
        def __init__(self, csv_file, split="PublicTest", transform=None):
            # 原仓库使用h5py加载预处理后的数据，这里保持csv解析但严格对齐处理逻辑
            self.df = pd.read_csv(csv_file)
            self.df = self.df[self.df["Usage"] == split]
            self.transform = transform
            self.image_size = image_size

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # 1. 像素解析与原仓库fer.py完全一致
            img = self.df.iloc[idx]['pixels']
            img = np.array([int(p) for p in img.split()], dtype=np.uint8)
            img = img.reshape((self.image_size, self.image_size))  # 48x48单通道
            
            # 2. 转为3通道（原仓库通过拼接单通道实现）
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)  # 形状为(48,48,3)
            img = Image.fromarray(img)  # 转为PIL Image
            
            # 3. 应用变换
            if self.transform:
                img = self.transform(img)
            
            label = self.df.iloc[idx]['emotion']
            return img, label

    # 数据集路径
    csv_path = os.path.join(root, "fer2013.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据集文件: {csv_path}")

    # 4. 预处理管道（严格复刻原仓库mainpro_FER.py）
    # 训练集变换（带数据增强）
    transform_train = transforms.Compose([
        transforms.RandomCrop(44),  # 关键：原仓库固定裁剪为44x44
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor，像素值归一化到[0,1]
    ])
    
    # 测试集变换（10折裁剪，不在此处平均，保持原仓库格式）
    transform_test = transforms.Compose([
        transforms.TenCrop(44),  # 生成10个44x44的裁剪区域
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    # 创建数据集
    train_dataset = FER2013Dataset(
        csv_file=csv_path,
        split="Training",
        transform=transform_train
    )
    test_dataset = FER2013Dataset(
        csv_file=csv_path,
        split="PrivateTest",
        transform=transform_test
    )

    # 5. DataLoader参数严格匹配（num_workers=1，batch_size原仓库用128）
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # 原仓库默认128
        shuffle=True,
        num_workers=4,  # 关键：原仓库固定1个worker
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,  # 原仓库默认128
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader

def load_fer2013_bk(batch_size=64, image_size=48, root="/home/ivan/XDU/code/Facial-Expression-Recognition.Pytorch-master/data"):
    """
    加载FER2013表情识别数据集，使用与原仓库一致的预处理方式
    原仓库地址：https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
    """
    class FER2013Dataset(Dataset):
        def __init__(self, csv_file, split="PublicTest", transform=None):
            self.df = pd.read_csv(csv_file)
            # 严格匹配原仓库的数据集划分方式
            self.df = self.df[self.df["Usage"] == split]
            self.transform = transform
            self.image_size = image_size

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            # 解析像素值（1通道灰度图）
            img = self.df.iloc[idx]['pixels']
            img = [int(p) for p in img.split()]
            img = torch.tensor(img, dtype=torch.float32).reshape(1, self.image_size, self.image_size)
            
            # 归一化到[0,1]范围（与原仓库一致）
            img = img / 255.0
            
            # 应用预处理
            if self.transform:
                img = self.transform(img)
            
            label = self.df.iloc[idx]['emotion']
            return img, label

    # 数据集路径
    csv_path = os.path.join(root, "fer2013.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据集文件: {csv_path}")

    # 预处理管道：保持与原仓库一致
    transform = transforms.Compose([
        # 原仓库使用的标准化参数
        transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 转换为3通道（如果模型需要）- 保持与原训练一致
        
    ])

    # 创建训练集和测试集
    train_dataset = FER2013Dataset(
        csv_file=csv_path,
        split="Training",  # 训练集划分
        transform=transform
    )
    test_dataset = FER2013Dataset(
        csv_file=csv_path,
        split="PrivateTest",  # 测试集划分（与权重对应）
        transform=transform
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱顺序
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader