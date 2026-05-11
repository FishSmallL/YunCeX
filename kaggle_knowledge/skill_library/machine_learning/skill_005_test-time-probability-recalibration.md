---
name: test-time-probability-recalibration
description: 对多模型集成输出的类别概率进行后处理，通过重新加权使测试集上各类别的预期实例数相等，从而解决类别分布漂移或竞赛中平衡对数损失等指标的特殊要求。
problems_solved: 测试集类别分布未知、类别不平衡学习、预测概率校准、balanced log loss优化
keyword: machine learning
category: 集成策略
source_kernel: 1_ICR_adv_model
source_competition: icr-identify-age-related-conditions
created: 2026-05-11
---

# Test Time Probability Recalibration

## 可解决的问题
测试集类别分布未知、类别不平衡学习、预测概率校准、balanced log loss优化

## 技巧说明
首先计算所有样本在各类别上的预测概率之和，作为该类别的估计实例数（class_est）。然后对每个样本的原始概率向量乘以逐类别的权重因子（1/class_est），最后按行归一化（除以新向量的和），确保概率和仍为1。这样，每个类别对最终预测的总概率贡献被平等化，相当于强行使模型在测试集上输出均匀的类别预期分布。当评价指标为平衡对数损失（balanced log loss）或要求对少数类同等重视时，这种后处理可大幅提升分数。它不改变模型本身的排序能力，仅针对评估指标调整概率尺度，是一种简单有效的测试时适应方法。

## 适用场景
竞赛中评价指标为平衡对数损失（或需要各类别等权重的指标）且测试集标签分布未知；或在生产中已知测试数据分布与训练数据不同且需要校准。

## 代码模式
```python
import numpy as np

# y_pred: shape (n_samples,) for binary probability of class 1
p1 = y_pred
p0 = 1 - p1
p = np.stack([p0, p1], axis=1)  # shape (n, 2)
class_0_est = p[:,0].sum()
class_1_est = p[:,1].sum()
# adjust
new_p = p * np.array([[1.0/class_0_est, 1.0/class_1_est]])
new_p = new_p / new_p.sum(axis=1, keepdims=True)
```

## 注意事项
该方法假设模型预测的概率具有正确的排序，仅需调整尺度；如果测试集真实分布极端不平衡，强制均匀化会损害实际性能；仅适用于概率输出，不能用于硬标签投票；调参时需注意浮点精度防止除零。
