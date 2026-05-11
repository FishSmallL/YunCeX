---
name: missing-value-extreme-fill
description: 使用列最小值减去最大值作为填充值处理存在缺失的数值列，同时用中位数填充无缺失的列，从而保留缺失值本身的模式信息，适用于表格数据的二分类或多分类任务。
problems_solved: 缺失值处理、缺失模式保留、高缺失率数值特征利用
keyword: machine learning
category: 数据预处理
source_kernel: 1_ICR_adv_model
source_competition: icr-identify-age-related-conditions
created: 2026-05-11
---

# Missing Value Extreme Fill

## 可解决的问题
缺失值处理、缺失模式保留、高缺失率数值特征利用

## 技巧说明
首先确定DataFrame中每列是否存在缺失值（bool Series），然后将该布尔序列乘以对应列的最小值减最大值，得到：有缺失的列为（min - max），一个远离正常范围的极端值；无缺失的列为0。然后将所有0替换为该列的中位数。最后用该Series填充原始DataFrame。这样做使缺失值被替换为明显区别于正常分布的极值，模型能够学习到“缺失”这一状态；同时无缺失的列使用中位数填充，避免极值干扰。相比简单的均值/中位数填充，该方法保留了缺失信息，有助于提升模型对缺失机制的拟合能力；相比创建缺失指示变量，该方法无需增加特征维度，且直接在数值空间中给出强信号。

## 适用场景
表格数据中存在部分数值列缺失，且缺失本身可能具有预测意义（非完全随机缺失）时；通常用于梯度提升树或深度学习模型。

## 代码模式
```python
import pandas as pd
import numpy as np

# assumes train_df is a DataFrame with numeric columns
nan_fill = train_df.isna().any()
nan_fill *= train_df.min() - train_df.max()
nan_fill[nan_fill == 0] = train_df.median()
train_df = train_df.fillna(nan_fill)
```

## 注意事项
特征必须为数值类型；若列的标准差极大，极值可能过于极端，应考虑缩放；当缺失比例极低时效果不明显；若所有列均无缺失，则等价于中位数填充，不会引入问题。
