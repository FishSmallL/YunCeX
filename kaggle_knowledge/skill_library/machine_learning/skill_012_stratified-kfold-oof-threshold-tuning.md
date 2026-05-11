---
name: stratified-kfold-oof-threshold-tuning
description: 利用分层 K 折交叉验证产生的包外预测概率，通过细粒度线搜索获得最大化准确率（或其他指标）的分类阈值，替代默认的 0.5 分割点。
problems_solved: 默认分类阈值不适应当前类别分布、模型校准不佳导致概率偏差、在特定评估指标下需要调整决策边界
keyword: machine learning
category: 调试技巧
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Stratified Kfold Oof Threshold Tuning

## 可解决的问题
默认分类阈值不适应当前类别分布、模型校准不佳导致概率偏差、在特定评估指标下需要调整决策边界

## 技巧说明
首先将数据集用 StratifiedKFold 划分，训练时保留每个验证折的概率预测组合成完整的 OOF 概率数组。然后定义一个阈值搜索函数，在 [0.05, 0.95] 范围内以步长 0.005 遍历，计算每个阈值对应的准确率，返回最优阈值。此过程在不引入额外验证集的情况下充分利用了全部训练数据评估模型，并找出最适合当前指标的分界点。与固定 0.5 相比，阈值调优能够弥补模型预测概率分布偏移或类不平衡带来的负面影响，对于以准确率为导向的二分类任务，往往能提升 0.5%~1% 的最终得分，且实现成本极低。

## 适用场景
在二分类问题中使用概率输出模型，且最终决策需要转化为硬标签的场景；特别适合评估指标为准确率、F1 等依赖于阈值的任务。

## 代码模式
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def tune_threshold(y_true, proba):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_a = 0.5, -1.0
    for t in ths:
        a = accuracy_score(y_true, (proba >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t, best_a

# 假设已有 oof 概率
best_t, best_a = tune_threshold(y, oof_proba)
```

## 注意事项
阈值搜索必须完全基于训练数据的 OOF 或独立验证集，绝不可使用测试集；若对标 F1 等非对称指标，需要修改内部评价函数；极端类不平衡时，网格范围可能需要缩小或改用概率分布百分位点进行搜索；若使用集成策略，应联合搜索权重与阈值。
