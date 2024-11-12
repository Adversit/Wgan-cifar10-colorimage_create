# Wgan-cifar10-colorimage_create
利用WGAN（Wasserstein Generative Adversarial Network）在CIFAR-10数据集上训练模型，实现从随机噪声生成高质量的彩色图像。包含实现代码、训练脚本、数据集处理及生成图像示例。

## 📁 项目结构

```
image-merger/
|—— data                # 存在下载的cifar10数据集
|—— models              # 存在保存的模型
|—— result
    |—— eval             # 存放生成的图片及IS，FID，KID三个指标
    |—— logs             # 存放使用tensorboard的日志
    |—— train            # 存放模型训练过程中每经过20个epoch生成的图片
├── model.py                # 深度学习模型定义和相关函数
|—— dataset.py             # 加载数据集
├── config.py                # 参数设置
├── requirements.txt       # 项目依赖
├── train.py              # 训练
|—— evaluate.py           # 评价
├── merge_images.py          # 处理图片合并的核心逻辑
├── structure.py              # 布局结构解析器，使用tensorboard查看模型结构
|── README.md            # 项目说明
             # 许可证文件
```

## 🚀 快速开始
### 环境要求

- Python 3.6+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM
### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/image-merger.git
cd image-merger
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

### 使用示例

1. 训练模型
```bash
python train.py
```
2. 评价模型
```bash
train evaluate.py
```
3. 合并图片
 ```bash
train merge_images.py 
```
4. 输出模型架构
```bash
train structure.py
``` 
## 📚 详细文档

## 📅 开发计划

- [ ] 支持更多图片格式
- [ ] 添加GUI界面
- [ ] 优化性能
- [ ] 添加批处理功能

## 🤝 贡献指南

1. Fork 本仓库
2. 创建新的分支 `git checkout -b feature/AmazingFeature`
3. 提交更改 `git commit -m 'Add some AmazingFeature'`
4. Push 到分支 `git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 📄 许可证

该项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件




---

如果这个项目对你有帮助，欢迎给一个 ⭐️
