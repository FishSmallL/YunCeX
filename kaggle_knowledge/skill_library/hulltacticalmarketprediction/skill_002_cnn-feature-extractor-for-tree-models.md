---
name: cnn-feature-extractor-for-tree-models
description: 利用一维卷积神经网络从原始时序数据中提取高阶表示，将其作为新的特征输入树模型（如 LGBM、XGBoost），实现深度学习与梯度提升树的融合。
use_case: 当原始时间序列信噪比低，手工特征难以捕获复杂模式时使用。适用于资产回报预测、传感器故障检测等时序回归/分类问题。要求训练数据量足够（至少数万序列样本），且序列长度相对固定。
keyword: Hull-Tactical-Market
category: 特征与表示
competition_type: 时序预测 | 表格回归
estimated_impact: 高
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# Cnn Feature Extractor For Tree Models

## 用途与场景
当原始时间序列信噪比低，手工特征难以捕获复杂模式时使用。适用于资产回报预测、传感器故障检测等时序回归/分类问题。要求训练数据量足够（至少数万序列样本），且序列长度相对固定。

## 技巧说明
构建一个 CNN 模型，输入形状为 (batch, 时间步, 特征数) 的序列。网络包含 1x1 卷积进行特征融合，堆叠卷积块提取模式，最后使用自适应池化将变长序列压缩为固定尺寸向量，再接全连接层输出指定维度的特征向量（例如 10 维）。训练时使用原始目标（如未来收益）和自定义损失（如夏普损失）优化网络，然后取全连接层的输出作为特征，与原始统计特征拼接，共同输入到 LightGBM、XGBoost 等树模型。相比仅手工特征，CNN 能自动学习短期交互和形态模式，弥补树模型对时序建模的不足。

## 代码模式
```python
import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, sequence_length, num_features, config):
        super().__init__()
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=config.kernel_size, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=config.kernel_size, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 10)  # 输出10维特征

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        x = self.feature_fusion(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
```

## 注意事项
CNN 特征维度不宜过高，否则树模型可能过拟合；提取的特征需与原始特征对齐时间索引；训练 CNN 时应使用验证集监控以防止过拟合，并尝试早停。
