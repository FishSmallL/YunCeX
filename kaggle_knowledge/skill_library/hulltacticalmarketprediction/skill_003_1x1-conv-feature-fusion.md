---
name: 1x1-conv-feature-fusion
description: 在时序卷积网络的首层使用 1x1 卷积核，对多变量时间序列进行跨通道的信息融合与降维，提升后续卷积层的效率。
use_case: "多变量时间序列建模，特别是当变量之间存在潜在交互或需要降维以减少参数时。适用于 TCN、CNN-LSTM 等架构的前端。"
keyword: Hull-Tactical-Market
category: 模型设计
competition_type: 时序预测 | 表格回归
estimated_impact: 中
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# 1X1 Conv Feature Fusion

## 用途与场景
多变量时间序列建模，特别是当变量之间存在潜在交互或需要降维以减少参数时。适用于 TCN、CNN-LSTM 等架构的前端。

## 技巧说明
对于形状为 (batch_size, channels, sequence_length) 的输入，在第一个卷积层采用 kernel_size=1 的 Conv1d 层，将原始的 channels 数量映射到固定维度（如 64）。1x1 卷积本质上对每个时间点的所有特征进行线性组合，相当于在时间维度上独立应用全连接层。其作用是融合不同特征间的关联信息，并降低通道维数，从而减少后续大核卷积的计算量。该结构通常接 BatchNorm 和激活函数，构成高效的特征预处理模块。相比于直接使用大核卷积，1x1 卷积先融合再提取时序模式，既能保留时间结构，又能增强模型对多变量关系的建模能力。

## 代码模式
```python
import torch
import torch.nn as nn

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        return self.fusion(x)
```

## 注意事项
当原始特征数较少（<10）时效果不明显；若特征已高度独立，1x1 卷积可能引入不必要的参数，可改用简单线性层或跳过。
