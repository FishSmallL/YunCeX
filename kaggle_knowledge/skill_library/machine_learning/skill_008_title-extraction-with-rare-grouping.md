---
name: title-extraction-with-rare-grouping
description: 从姓名等字符串中提取身份头衔，将出现频次极低的头衔归入同一“稀有”类别，在保留社会身份信息的同时大幅降低类别基数，防止过拟合。
problems_solved: 高基数文本特征编码、稀有类别导致过拟合、从非结构化姓名中提取社会地位信号、提升模型健壮性
keyword: machine learning
category: 特征工程
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Title Extraction With Rare Grouping

## 可解决的问题
高基数文本特征编码、稀有类别导致过拟合、从非结构化姓名中提取社会地位信号、提升模型健壮性

## 技巧说明
使用正则表达式从姓名列抽取出头衔（如 Mr、Miss、Dr），然后利用 `.replace()` 将少量出现（例如少于 10 次）的头衔统一映射为 'Rare'，并填补缺失。这样既保留了高频头衔的社会地位信号，又将小众头衔归并，避免模型针对极少样本学习到噪声规律。相比直接使用原始姓名或高基数类别，该方法在类别爆炸时能显著减少模型复杂度，并利用头衔与目标变量的稳定关联提升泛化能力。结合词干化或同义映射（如 Mlle→Miss），还能进一步合并语义近似的类别。

## 适用场景
数据中存在姓名、称呼、职位等字符串列，且目标变量与社会身份明显相关时使用，适用于树模型、线性模型（后续需编码）以及深度学习嵌入层。

## 代码模式
```python
import pandas as pd
import numpy as np

df = df.copy()
title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
title = title.replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
})
vc = title.value_counts()
rare = vc[vc < 10].index
title = title.replace(rare, "Rare")
df["Title"] = title.fillna("Rare")
```

## 注意事项
频次阈值（如10）需要根据总样本量调整，过小无法合并足够类别，过大会丢失重要区别；测试集可能出现未见过的头衔，归入'Rare'时需保持与训练逻辑一致；某些场景下头衔与隐私敏感字段相关，应考虑合规性。
