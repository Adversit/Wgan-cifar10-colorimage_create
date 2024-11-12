import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config
from model import Generator, Discriminator

def visualize_model_structure():
    # 初始化配置
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    netG = Generator(config).to(device)
    netD = Discriminator(config).to(device)
    
    # 创建示例输入
    dummy_noise = torch.randn(1, config.latent_dim).to(device)  # 生成器输入
    dummy_image = torch.randn(1, 3, config.image_size, config.image_size).to(device)  # 判别器输入
    
    # 创建TensorBoard writer
    writer = SummaryWriter('result/logs/model_structure')
    
    # 添加模型图
    try:
        # 可视化生成器结构
        writer.add_graph(netG, dummy_noise)
        print("生成器结构已保存")
        
        # 可视化判别器结构
        writer.add_graph(netD, dummy_image)
        print("判别器结构已保存")
        
    except Exception as e:
        print(f"模型可视化时出错: {str(e)}")
    
    # 添加模型参数统计
    total_params_G = sum(p.numel() for p in netG.parameters())
    total_params_D = sum(p.numel() for p in netD.parameters())
    
    writer.add_text('Model/Generator_Params', f'Total parameters: {total_params_G:,}')
    writer.add_text('Model/Discriminator_Params', f'Total parameters: {total_params_D:,}')
    
    # 关闭writer
    writer.close()
    
    print(f"\n模型结构已保存到 result/logs/model_structure")
    print(f"生成器参数量: {total_params_G:,}")
    print(f"判别器参数量: {total_params_D:,}")
    print("\n使用以下命令启动TensorBoard查看模型结构：")
    print("tensorboard --logdir=result/logs/model_structure")

if __name__ == '__main__':
    visualize_model_structure()
