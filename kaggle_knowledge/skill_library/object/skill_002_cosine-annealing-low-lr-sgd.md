---
name: cosine-annealing-low-lr-sgd
description: "采用余弦退火调度配合极低初始学习率1e-4与SGD优化器，实现大型检测模型在合成数据上的平稳收敛，防止剧烈振荡和过拟合。"
use_case: 使用大型检测模型在合成或小规模目标检测数据集上进行微调，需要避免过拟合和训练不稳定的场景。
keyword: object
category: 训练策略
competition_type: 目标检测
estimated_impact: 高
source_kernel: adl212
source_competition: synthetic-2-real-object-detection-challenge
created: 2026-05-14
---

# Cosine Annealing Low Lr Sgd

## 用途与场景
使用大型检测模型在合成或小规模目标检测数据集上进行微调，需要避免过拟合和训练不稳定的场景。

## 技巧说明
设置cos_lr=True启用余弦退火学习率调度，lr0=0.0001作为初始学习率，optimizer='SGD'，momentum=0.975，weight_decay=0.0001。余弦退火使学习率从初始值平滑地周期性下降，无突然断崖，有助于跳出尖锐局部极小值并收敛到更平坦的泛化区域。极低初始学习率1e-4避免了在有限且可能存在域差异的合成数据上快速过拟合；SGD优化器自带噪声，相比Adam等自适应方法具有更好的泛化能力。该组合尤其适合从预训练权重开始微调大型模型（如YOLOv11x）的场景。

## 代码模式
```python
results = model.train(
    data='yolo_params.yaml',
    epochs=100,
    cos_lr=True,
    lr0=0.0001,
    optimizer='SGD',
    momentum=0.975,
    weight_decay=0.0001,
    ...
)
```

## 注意事项
极低学习率会延缓收敛，需要足够多的epochs（配合早停）；可适当增大warmup_epochs缓解初始震荡；若数据集较大可尝试提高lr0至1e-3，但需监控验证曲线。
