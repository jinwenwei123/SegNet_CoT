"""
此处设置所有参数
"""
model_opt = {
    "state": "image",  # train or test or image ,选择模式
    "device": "cuda",  # 训练设备
    "img_size": (320, 480),  # 图片大小
    "attention": False,  # 是否启用CoTAttention模块
    "num_classes": 20 + 1,  # 类别数 + 背景
    "batch_size": 6,  # 样本簇大小
    "lr": 0.001,  # 学习率
    "epoch": 1000,  # 训练轮数
    "root": "./dataset",  # 数据集根目录
    "pretrain": True,  # 是否启用预训练
    "weight_save_path": "./weight",  # 权重文件目录
    "model": "model_S_epoch100.pth",  # 模型名
    "predictions_path": "./predictions",  # 预测图片目录
}
