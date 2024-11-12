import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset(config):
    """获取CIFAR10训练集用于训练"""
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,  # 明确使用训练集
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader 