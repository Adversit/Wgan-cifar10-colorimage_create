import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import pandas as pd
import json
import warnings

# 忽略TypedStorage警告
warnings.filterwarnings(
    "ignore", 
    message="TypedStorage is deprecated"
)

from config import Config
from model import Generator

def prepare_real_samples(config):
    """准备真实样本用于评估，使用CIFAR10测试集"""
    real_dir = os.path.join(config.eval_dir, 'real_samples')
    os.makedirs(real_dir, exist_ok=True)
    
    # 加载CIFAR10测试集
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=False,  # 使用测试集
        download=True,
        transform=transform
    )
    
    # 从测试集中随机选择680张图像
    indices = torch.randperm(len(dataset))[:680]
    
    # 保存选中的图像
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(real_dir, f'real_{i}.png'))
    
    return real_dir

def calculate_all_metrics(real_path, fake_path):
    """计算FID、IS和KID指标"""
    try:
        # 计算所有指标
        metrics = calculate_metrics(
            input1=real_path,
            input2=fake_path,
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size=680,
            verbose=False,
        )
        
        return {
            'fid': metrics['frechet_inception_distance'],
            'is_mean': metrics['inception_score_mean'],
            'is_std': metrics['inception_score_std'],
            'kid_mean': metrics['kernel_inception_distance_mean'],
            'kid_std': metrics['kernel_inception_distance_std'],
        }
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None

def evaluate():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    os.makedirs(config.eval_dir, exist_ok=True)
    
    # 准备真实样本
    real_samples_dir = prepare_real_samples(config)
    
    # 初始化TensorBoard
    writer = SummaryWriter(os.path.join(config.log_dir, 'eval'))
    
    # 用于存储所有epoch的指标
    all_metrics = []
    
    # 遍历所有保存的模型
    for epoch in range(config.save_interval, config.num_epochs + 1, config.save_interval):
        print(f"\n评估 Epoch {epoch} 的模型...")
        
        # 创建当前epoch的评估目录
        eval_epoch_dir = os.path.join(config.eval_dir, f'epoch_{epoch}')
        os.makedirs(eval_epoch_dir, exist_ok=True)
        
        # 加载模型
        try:
            checkpoint = torch.load(
                f'{config.model_dir}/model_epoch_{epoch}.pth',
                map_location=device
            )
            netG = Generator(config).to(device)
            netG.load_state_dict(checkpoint['generator_state_dict'])
            netG.eval()
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            continue
        
        # 生成图像
        print("生成评估用图像...")
        with torch.no_grad():
            remaining_images = 680
            for i in tqdm(range(0, 680, config.eval_batch_size)):
                batch_size = min(config.eval_batch_size, remaining_images)
                noise = torch.randn(batch_size, config.latent_dim, device=device)
                fake_imgs = netG(noise)
                
                for j, img in enumerate(fake_imgs):
                    img = (img.cpu().clone() + 1) / 2.0
                    img = transforms.ToPILImage()(img)
                    img.save(os.path.join(eval_epoch_dir, f'img_{i+j}.png'))
                
                remaining_images -= batch_size
        
        # 计算评估指标
        print("计算评估指标...")
        metrics = calculate_all_metrics(real_samples_dir, eval_epoch_dir)
        
        if metrics is not None:
            # 添加epoch信息到指标字典
            metrics['epoch'] = epoch
            all_metrics.append(metrics)
            
            # 记录到TensorBoard
            writer.add_scalar('Metrics/FID', metrics['fid'], epoch)
            writer.add_scalar('Metrics/IS_mean', metrics['is_mean'], epoch)
            writer.add_scalar('Metrics/KID_mean', metrics['kid_mean'], epoch)
            
            # 保存当前epoch的评估结果
            metrics_file = os.path.join(eval_epoch_dir, 'metrics.txt')
            metrics_json = os.path.join(eval_epoch_dir, 'metrics.json')
            
            # 保存为文本文件
            with open(metrics_file, 'w') as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"FID: {metrics['fid']:.4f}\n")
                f.write(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}\n")
                f.write(f"KID: {metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}\n")
            
            # 保存为JSON文件
            with open(metrics_json, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"Epoch {epoch} 评估完成:")
            print(f"FID: {metrics['fid']:.4f}")
            print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
            print(f"KID: {metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}")
    
    # 保存所有epoch的指标到CSV文件
    if all_metrics:
        metrics_csv = os.path.join(config.eval_dir, 'all_metrics.csv')
        pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
        
        # 生成汇总报告
        summary_file = os.path.join(config.eval_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("评估汇总\n")
            f.write("========\n\n")
            
            # 找出最佳分数
            best_fid = min(m['fid'] for m in all_metrics)
            best_fid_epoch = next(m['epoch'] for m in all_metrics if m['fid'] == best_fid)
            f.write(f"最佳 FID 分数: {best_fid:.4f} (Epoch {best_fid_epoch})\n")
            
            best_is = max(m['is_mean'] for m in all_metrics)
            best_is_epoch = next(m['epoch'] for m in all_metrics if m['is_mean'] == best_is)
            f.write(f"最佳 IS 分数: {best_is:.4f} (Epoch {best_is_epoch})\n")
            
            best_kid = min(m['kid_mean'] for m in all_metrics)
            best_kid_epoch = next(m['epoch'] for m in all_metrics if m['kid_mean'] == best_kid)
            f.write(f"最佳 KID 分数: {best_kid:.4f} (Epoch {best_kid_epoch})\n")
    
    writer.close()
    print("\n评估完成！结果已保存到", config.eval_dir)

if __name__ == '__main__':
    evaluate()