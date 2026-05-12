---
name: heterogeneous-model-weighted-ensemble
description: 将梯度提升树、线性模型和神经网络等异构模型按固定权重融合，综合不同模型对数据的不同假设以提升泛化性能。
use_case: 适用于表格数据竞赛中需要稳定最终提交得分的场景，特别是当单个模型排名波动较大时。要求验证集能够准确反映未来分布，以便合理分配权重。
keyword: Hull-Tactical-Market
category: 集成与后处理
competition_type: 表格回归 | 时序预测
estimated_impact: 中
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# Heterogeneous Model Weighted Ensemble

## 用途与场景
适用于表格数据竞赛中需要稳定最终提交得分的场景，特别是当单个模型排名波动较大时。要求验证集能够准确反映未来分布，以便合理分配权重。

## 技巧说明
分别训练 LightGBM、XGBoost 和 ElasticNet 三个模型，其中树模型使用包含 CNN 特征的全部特征集，线性模型仅使用原始基础特征（以避免过拟合和共线性）。在推理时，对每个模型单独预测，然后按预定义权重（如 {'lgb': 0.4, 'xgb': 0.4, 'enet': 0.2}）计算加权平均，得到原始预测；再结合后处理（如波动率缩放）输出最终信号。这种异构集成利用了树模型对非线性交互的优势、正则化线性模型的稳定性以及 CNN 的表示能力，显著降低单一模型的风险。

## 代码模式
```python
import numpy as np

def ensemble_predict(models, weights, X_ens, X_base):
    preds = {}
    preds['lgb'] = models['lgb'].predict(X_ens)
    preds['xgb'] = models['xgb'].predict(X_ens)
    preds['enet'] = models['enet'].predict(X_base)
    raw_pred = sum(preds[name] * weights[name] for name in models)
    return raw_pred
```

## 注意事项
权重最好基于验证集上的性能进行优化（如网格搜索或 Optuna）；避免在训练数据上挑选权重以防过拟合；异构模型的特征对齐需提前规划，确保推理时特征可用。
