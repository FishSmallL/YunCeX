---
name: weighted-blending-with-joint-threshold-optimization
description: 将多个模型的包外概率进行加权平均，并联合搜索最佳权重和分类阈值以最大化评估指标，实现细粒度集成与决策边界调优。
problems_solved: 多模型方差缩减、类别不平衡下的阈值选择、单一模型输出概率校准不足、集成权重依赖经验时容易次优
keyword: machine learning
category: 集成策略
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Weighted Blending With Joint Threshold Optimization

## 可解决的问题
多模型方差缩减、类别不平衡下的阈值选择、单一模型输出概率校准不足、集成权重依赖经验时容易次优

## 技巧说明
先通过分层 K 折交叉验证生成每个模型的 OOF 概率，再对 CatBoost 和 LightGBM 的 OOF 概率进行线性组合 `p = w * proba_cb + (1-w) * proba_lgb`。在 [0,1] 区间按步长 0.01 遍历权重 w，对每个 w 调用阈值搜索函数在 0.05–0.95 之间寻找使准确率最大的阈值 t，最后选取 OOF 准确率最高的 (w, t) 配对。推理时用同样的权重合成测试集概率，并以 t 产生最终硬预测。该方法同时优化了模型融合的权重分配和针对特定指标的决策阈值，比起固定 0.5 阈值或平均权重，通常能显著提升最终指标，尤其当不同模型对类的偏向不一致时效果更明显。

## 适用场景
拥有两个或以上异质模型（如树模型与神经网络、不同梯度提升库），且验证指标对阈值敏感时使用；适用于二分类竞赛或业务场景中需最大化准确率、F1 或特定利润函数的情况。

## 代码模式
```python
import numpy as np
from sklearn.metrics import accuracy_score

def tune_threshold(y_true, proba):
    ths = np.linspace(0.05, 0.95, 181)
    best_t, best_a = 0.5, -1.0
    for t in ths:
        a = accuracy_score(y_true, (proba >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t, best_a

best = {"acc": -1.0, "w": 0.5, "t": 0.5}
for w in np.linspace(0, 1, 101):
    p = w * oof_cb + (1 - w) * oof_lgb
    t, a = tune_threshold(y, p)
    if a > best["acc"]:
        best = {"acc": a, "w": float(w), "t": float(t)}
w_opt, t_opt = best["w"], best["t"]
test_blend = w_opt * test_cb + (1 - w_opt) * test_lgb
preds = (test_blend >= t_opt).astype(int)
```

## 注意事项
网格搜索的精确度取决于步长，过细会导致过配于验证数据；如果测试分布与验证分布存在偏移，联合优化可能过拟合 OOF 的性能，需留出独立 hold-out 集进行最终评估；对于严重类别不平衡问题，应将 accuracy 替换为更合适的指标（如 F1、MCC）进行搜索。
