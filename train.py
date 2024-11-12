import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import torch.autograd as autograd
import glob

from config import Config
from dataset import get_dataset
from model import Generator, Discriminator

def find_latest_model():
    """查找最后保存的模型"""
    # 确保使用正确的目录路径
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None, 0
    
    # 查找所有模型文件
    model_files = glob.glob(os.path.join(model_dir, 'model_epoch_*.pth'))
    if not model_files:
        return None, 0
    
    # 从文件名中提取epoch数
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    latest_epoch = max(epochs)
    latest_model = os.path.join(model_dir, f'model_epoch_{latest_epoch}.pth')
    
    print(f"找到最新的模型文件: {latest_model}")
    return latest_model, latest_epoch

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """计算梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建保存目录
    os.makedirs(config.train_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # 初始化模型
    netG = Generator(config).to(device)
    netD = Discriminator(config).to(device)
    
    # 初始化优化器
    optimG = torch.optim.Adam(netG.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
    optimD = torch.optim.Adam(netD.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
    
    # 查找最新的模型文件
    latest_model, start_epoch = find_latest_model()
    if latest_model:
        print(f"从epoch {start_epoch} 继续训练...")
        checkpoint = torch.load(latest_model, map_location=device)
        netG.load_state_dict(checkpoint['generator_state_dict'])
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        optimG.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimD.load_state_dict(checkpoint['optimizer_d_state_dict'])
    else:
        print("从头开始训练...")
        start_epoch = 0
    
    # 初始化学习率调度器
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimG, 
        step_size=config.lr_decay_every,
        gamma=config.lr_decay_rate,
        last_epoch=start_epoch-1
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optimD,
        step_size=config.lr_decay_every,
        gamma=config.lr_decay_rate,
        last_epoch=start_epoch-1
    )
    
    # 获取数据加载器
    dataloader = get_dataset(config)
    
    # 初始化TensorBoard
    writer = SummaryWriter(config.log_dir)
    
    # 训练循环
    for epoch in range(start_epoch, config.num_epochs):
        pbar = tqdm(dataloader)
        for i, (real_imgs, _) in enumerate(pbar):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # 训练判别器
            for _ in range(config.d_steps):
                netD.zero_grad()
                
                # 真实图像的损失
                d_real = netD(real_imgs)
                d_real_loss = -d_real.mean()
                
                # 生成图像的损失
                noise = torch.randn(batch_size, config.latent_dim, device=device)
                fake_imgs = netG(noise)
                d_fake = netD(fake_imgs.detach())
                d_fake_loss = d_fake.mean()
                
                # 梯度惩罚
                gradient_penalty = compute_gradient_penalty(
                    netD, real_imgs, fake_imgs.detach(), device
                )
                
                # 总损失
                d_loss = d_real_loss + d_fake_loss + config.gp_lambda * gradient_penalty
                d_loss.backward()
                optimD.step()
            
            # 训练生成器
            netG.zero_grad()
            fake_imgs = netG(noise)
            g_loss = -netD(fake_imgs).mean()
            g_loss.backward()
            optimG.step()
            
            # 更新进度条
            pbar.set_description(
                f"Epoch [{epoch}/{config.num_epochs}] "
                f"d_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f} "
                f"lr_g: {scheduler_g.get_last_lr()[0]:.6f}"
            )
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('LR/Generator', scheduler_g.get_last_lr()[0], epoch)
            writer.add_scalar('LR/Discriminator', scheduler_d.get_last_lr()[0], epoch)
        
        # 学习率调度
        if epoch >= config.lr_decay_start:
            scheduler_g.step()
            scheduler_d.step()
        
        # 保存模型和生成的样本
        if (epoch + 1) % config.save_interval == 0:
            checkpoint = {
                'generator_state_dict': netG.state_dict(),
                'discriminator_state_dict': netD.state_dict(),
                'optimizer_g_state_dict': optimG.state_dict(),
                'optimizer_d_state_dict': optimD.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'{config.model_dir}/model_epoch_{epoch+1}.pth')
            
            with torch.no_grad():
                sample_noise = torch.randn(64, config.latent_dim, device=device)
                sample_imgs = netG(sample_noise)
                save_image(sample_imgs, f'{config.train_dir}/epoch_{epoch+1}.png',
                          normalize=True, nrow=8)
    
    writer.close()

if __name__ == '__main__':
    train()