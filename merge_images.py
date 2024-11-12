import os
import glob
from PIL import Image
import math
import torch
from torchvision.utils import make_grid
from torchvision import transforms
from tqdm import tqdm

def merge_images_batch(image_paths, save_path, n_cols=8, batch_size=64):
    """将一批图像合并为8x8网格"""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        images.append(img)
    
    # 将图像列表转换为tensor
    images = torch.stack(images)
    
    # 创建8x8网格
    grid = make_grid(images, nrow=n_cols, padding=2, normalize=False)
    
    # 保存合并后的图像
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(save_path)

def process_epoch_images(src_dir, grid_dir, epoch):
    """处理一个epoch目录下的所有图像"""
    # 获取所有图像路径
    image_paths = sorted(glob.glob(os.path.join(src_dir, 'img_*.png')))
    total_images = len(image_paths)
    
    # 计算需要生成多少个网格图片
    batch_size = 64  # 8x8网格
    num_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"\n处理 epoch_{epoch} 的图像 (共{total_images}张，将生成{num_batches}个网格)")
    
    # 为当前epoch创建保存目录
    epoch_grid_dir = os.path.join(grid_dir, f'epoch_{epoch}')
    os.makedirs(epoch_grid_dir, exist_ok=True)
    
    # 按批次处理图像
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        
        # 如果最后一批不足64张，则重复使用前面的图像填充
        batch_paths = image_paths[start_idx:end_idx]
        if len(batch_paths) < batch_size:
            batch_paths = batch_paths + batch_paths[:batch_size-len(batch_paths)]
        
        save_path = os.path.join(epoch_grid_dir, f'grid_{batch_idx+1}.png')
        
        try:
            merge_images_batch(batch_paths, save_path)
        except Exception as e:
            print(f"处理批次 {batch_idx+1} 时出错: {str(e)}")
            continue

def main():
    # 基础目录
    eval_dir = 'result/eval'
    
    # 创建保存网格图像的目录
    grid_dir = os.path.join(eval_dir, 'grid_images')
    os.makedirs(grid_dir, exist_ok=True)
    
    # 处理从epoch_20到epoch_200的所有目录
    print("开始处理图像...")
    for epoch in tqdm(range(20, 201, 20)):
        src_dir = os.path.join(eval_dir, f'epoch_{epoch}')
        if not os.path.exists(src_dir):
            print(f"跳过 epoch_{epoch}: 目录不存在")
            continue
        
        process_epoch_images(src_dir, grid_dir, epoch)
    
    print("\n处理完成！网格图像已保存到:", grid_dir)

if __name__ == '__main__':
    main() 