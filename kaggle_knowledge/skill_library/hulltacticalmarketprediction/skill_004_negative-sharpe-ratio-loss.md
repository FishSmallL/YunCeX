---
name: negative-sharpe-ratio-loss
description: 将投资组合的夏普比率直接作为模型优化目标，通过负夏普损失引导模型生成高收益、低波动的交易信号。
use_case: 适用于以夏普比率或类似风险调整指标为评估标准的金融时间序列竞赛，模型输出为连续信号（多/空强度）。也可推广到任何需要直接优化比率目标的回归任务。
keyword: Hull-Tactical-Market
category: 损失与指标
competition_type: 时序预测 | 金融回归
estimated_impact: 高
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# Negative Sharpe Ratio Loss

## 用途与场景
适用于以夏普比率或类似风险调整指标为评估标准的金融时间序列竞赛，模型输出为连续信号（多/空强度）。也可推广到任何需要直接优化比率目标的回归任务。

## 技巧说明
定义损失函数为负夏普比率：先将模型的原始预测 y_pred 通过阈值（如 >0）转换为仓位 signal（预测大于 0 时取预测值，否则为 0）；然后计算策略回报 strategy_returns = signal * y_true（实际未来收益）；再计算该序列的均值与标准差，得到 Sharpe = mean/(std + ε)；最后返回 -Sharpe 作为损失。在反向传播时，该损失直接推动模型提升风险调整后收益。相比均方误差（MSE），它更贴近金融竞赛的评估指标（如 Hull Tactical 的效用函数），并且能自动惩罚高波动策略。

## 代码模式
```python
import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # y_pred: 原始预测信号，y_true: 真实回报
        positions = torch.where(y_pred > 0, y_pred, torch.tensor(0.0, device=y_pred.device))
        strategy_returns = positions * y_true
        mean_return = torch.mean(strategy_returns)
        std_return = torch.std(strategy_returns)
        sharpe = mean_return / (std_return + self.epsilon)
        return -sharpe
```

## 注意事项
该损失严重依赖于批量样本的代表性，batch size 过小可能导致均值/标准差估计不稳定；建议每个 batch 包含至少数百个时间点；epsilon 防止除零；在验证时仍需使用原始评估指标。
