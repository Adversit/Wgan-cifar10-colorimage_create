class Config:
    # 数据集配置
    dataset = 'cifar10'
    image_size = 32
    num_classes = 10
    
    # 模型配置
    latent_dim = 128
    g_ch = 48
    d_ch = 48
    
    # 训练配置
    num_epochs = 500
    batch_size = 32
    lr_g = 2e-4
    lr_d = 2e-4
    beta1 = 0.0
    beta2 = 0.999
    
    # 学习率调度器配置
    lr_decay_start = 100  # 开始衰减的epoch
    lr_decay_rate = 0.5   # 衰减率
    lr_decay_every = 40   # 每隔多少个epoch衰减一次
    
    # 训练稳定性配置
    d_steps = 2           # 判别器训练步数
    gp_lambda = 10.0      # 梯度惩罚系数
    
    # 保存配置
    save_interval = 20
    eval_batch_size = 50
    n_samples = 64
    
    # 路径配置
    train_dir = 'result/train'
    eval_dir = 'result/eval'
    log_dir = 'result/logs'
    model_dir = 'models' 