---
name: ultra-low-confidence-threshold-for-full-pr
description: "推理时将置信度阈值降至0.001，保留几乎所有检测框以重建完整的Precision-Recall曲线，从而获得更准确的mAP评估。"
use_case: 目标检测竞赛的最终测试集推理与提交阶段，需要输出mAP等基于PR曲线的指标评估。
keyword: object
category: 验证与评估
competition_type: 目标检测
estimated_impact: 高
source_kernel: adl212
source_competition: synthetic-2-real-object-detection-challenge
created: 2026-05-14
---

# Ultra Low Confidence Threshold For Full Pr

## 用途与场景
目标检测竞赛的最终测试集推理与提交阶段，需要输出mAP等基于PR曲线的指标评估。

## 技巧说明
在模型预测阶段设置conf=0.001，使模型输出全部可能的检测框（包括极低置信度的候选），然后对所有框进行排序和匹配计算mAP。目标检测竞赛中评价指标（如mAP@0.5:0.95）要求尽可能完整的PR曲线，常规较高的置信度阈值（如0.25或0.5）会提前过滤掉一些低置信度的正确检测（高召回区域），导致曲线不完整，低估真实mAP。使用conf=0.001后再配合官方评估脚本，可以确保PR曲线覆盖从高精度到高召回的全部区域，尤其能捕获到部分遮挡或难分样本的正确检测。

## 代码模式
```python
results = model.predict(
    '/path/to/test/images',
    imgsz=640,
    conf=0.001,
    save=True,
    verbose=False
)
```

## 注意事项
仅适用于评估，实际部署需根据应用设置合适阈值（如0.25）；极低阈值会生成海量检测框，可能超出提交文件大小限制，需验证提交格式的容量；务必配合test-stage的评分代码使用，不要误用在带独立后处理的推理管道中。
