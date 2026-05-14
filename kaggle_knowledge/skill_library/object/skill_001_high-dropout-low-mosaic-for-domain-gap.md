---
name: high-dropout-low-mosaic-for-domain-gap
description: 通过联合调整Dropout和Mosaic增强概率，缓解合成数据训练到真实场景推理时的过拟合问题。
use_case: 训练数据为合成图像但测试数据为真实图像的目标检测任务，尤其是物体纹理和背景差异显著的竞赛场景。
keyword: object
category: 训练策略
competition_type: 目标检测
estimated_impact: 高
source_kernel: adl212
source_competition: synthetic-2-real-object-detection-challenge
created: 2026-05-14
---

# High Dropout Low Mosaic For Domain Gap

## 用途与场景
训练数据为合成图像但测试数据为真实图像的目标检测任务，尤其是物体纹理和背景差异显著的竞赛场景。

## 技巧说明
训练时设置dropout=0.4，远高于YOLO默认值，并设置mosaic=0.2，只以20%概率进行Mosaic增强。dropout=0.4强正则化全连接层，迫使模型学习更加鲁棒的特征表达，减少对合成域特定噪声的依赖。Mosaic增强原用于提升小目标和场景多样性，但在合成到真实域迁移任务中，过强的Mosaic（默认1.0）会引入大量合成拼接伪影，使模型过拟合合成域的虚假纹理和布局。将概率降至0.2保留了部分多尺度上下文信息，又避免过度破坏真实域的自然统计特性。两者联合使用，可显著缩小合成-真实域之间的泛化差距。

## 代码模式
```python
from ultralytics import YOLO
model = YOLO('yolo11x.yaml').load('yolo11x.pt')
results = model.train(
    data='yolo_params.yaml',
    epochs=100,
    imgsz=640,
    patience=20,
    dropout=0.4,
    mosaic=0.2,
    ...
)
```

## 注意事项
若真实域数据充足，可适当提高mosaic概率；dropout过高可能导致欠拟合，建议配合早停监控验证损失；Mosaic概率低于0.1时可能失去多尺度收益，需根据合成数据多样性调整。
