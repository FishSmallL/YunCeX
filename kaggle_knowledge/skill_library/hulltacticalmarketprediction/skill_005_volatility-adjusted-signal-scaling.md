---
name: volatility-adjusted-signal-scaling
description: 在产生最终预测信号时，根据最近一段时间的收益波动率对模型原始输出进行缩放，使策略在不同波动环境下保持风险一致。
use_case: 适用于任何需要将模型输出转化为实际仓位或信号强度的交易竞赛，特别是当评估指标奖励低波动策略时。也可用于非金融领域需要基于历史统计量归一化输出的场景。
keyword: Hull-Tactical-Market
category: 集成与后处理
competition_type: 时序预测 | 金融回归
estimated_impact: 中
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# Volatility Adjusted Signal Scaling

## 用途与场景
适用于任何需要将模型输出转化为实际仓位或信号强度的交易竞赛，特别是当评估指标奖励低波动策略时。也可用于非金融领域需要基于历史统计量归一化输出的场景。

## 技巧说明
维护一个滚动窗口（例如最近 20 步）的历史真实收益序列。在每次获得模型原始预测 raw_pred 后，计算窗口内收益的标准差 vol = np.std(recent_returns)；然后将最终信号定义为 raw_pred / (vol + ε)，或类似的比例缩放（如限制最大头寸）。这实际上是一种简单的波动率目标技术（volatility targeting），使策略在高波动期降低头寸，低波动期增加头寸，从而平滑净值曲线，降低极端波动风险。它独立于模型训练，实现成本极低。

## 代码模式
```python
import numpy as np

def apply_volatility_scaling(raw_pred, recent_returns, epsilon=1e-8, max_leverage=1.0):
    vol = np.std(recent_returns)
    if vol < epsilon:
        vol = 1.0  # 防止除零
    scaled_pred = raw_pred / vol
    # 可选：裁剪到最大杠杆
    scaled_pred = np.clip(scaled_pred, -max_leverage, max_leverage)
    return scaled_pred
```

## 注意事项
滚动窗口长度需与策略的回望期和目标波动率匹配；若市场发生结构性突变，历史波动率可能失效；可考虑使用指数加权标准差以提高响应速度。
