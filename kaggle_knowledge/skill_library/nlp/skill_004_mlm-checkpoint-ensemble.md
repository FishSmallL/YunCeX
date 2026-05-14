---
name: mlm-checkpoint-ensemble
description: 对多个不同预训练MLM模型（DeBERTa/Electra/RoBERTa）的logits进行加权平均，利用模型多样性提升选择题预测稳定性。
use_case: 解决单一MLM模型预测不稳定、泛化能力有限的问题；适用于任何需要提升多项选择准确率的分类场景。
keyword: nlp
category: 集成与后处理
competition_type: 自然语言处理-多项选择问答
estimated_impact: 高
source_kernel: Podpall_2
source_competition: kaggle-llm-science-exam
created: 2026-05-14
---

# Mlm Checkpoint Ensemble

## 用途与场景
解决单一MLM模型预测不稳定、泛化能力有限的问题；适用于任何需要提升多项选择准确率的分类场景。

## 技巧说明
在阅读理解阶段，分别训练/微调了基于DeBERTa、Electra、RoBERTa三种架构的多个检查点。推理时，每个模型各自对5个选项输出logits，先将同一架构的多个检查点求均值，得到该架构的logits向量，再按照预设权重（DEBERTA_WEIGHT=0.5, ELECTRA_WEIGHT=0.3, ROBERTA_WEIGHT=0.2）进行加权求和，得到最终logits。这种集成方式利用了不同预训练目标、不同mask策略带来的互补性，能有效降低单一模型过拟合或偏见的影响，提高最终预测的准确率和MAP。相比简单的硬投票，软加权融合能更好地捕捉模型置信度的差异，并可结合验证集灵活调整权重。

## 代码模式
```python
DEBERTA_WEIGHT = 0.5
ELECTRA_WEIGHT = 0.3
ROBERTA_WEIGHT = 0.2

# deberta_logits_result, electra_logits_result, roberta_logits_result 均为形状 [n_questions, 5] 的 tensor 列表
dberta_avg = torch.stack(deberta_logits_result).mean(dim=0)
electra_avg = torch.stack(electra_logits_result).mean(dim=0)
roberta_avg = torch.stack(roberta_logits_result).mean(dim=0)

final_logits = DEBERTA_WEIGHT * dberta_avg + ELECTRA_WEIGHT * electra_avg + ROBERTA_WEIGHT * roberta_avg
preds = torch.argsort(-final_logits, dim=1)[:, :3]  # 取top3
```

## 注意事项
权重应基于验证集的准确率（或交叉验证）来设置；若不同模型logits尺度差异较大，可先进行温度缩放再融合；检查点数量越多集成效果越好，但推理时间线性增加。
