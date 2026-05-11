---
name: cabin-deck-extraction
description: 从包含舱室编号的字符串中截取首字母作为甲板类别，将高位层级信息浓缩为低基数分类特征，同时缺失值作为单独类别保留。
problems_solved: 从半结构化字符串提取层级信息、处理缺失值时的信息保留、降低文本特征基数
keyword: machine learning
category: 特征工程
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Cabin Deck Extraction

## 可解决的问题
从半结构化字符串提取层级信息、处理缺失值时的信息保留、降低文本特征基数

## 技巧说明
对 Cabin 列取首个字符（通常代表甲板层）并赋给新列 Deck，缺失值填充为 'U'（Unknown）或 'None'，再转为类别类型。这种做法利用船舱编号的固有层级结构，将多元字符串压缩为少数几个甲板类别，同时将缺失值本身视为一种信息（可能意味着未记录或经济舱无固定舱室）。相比直接删除 Cabin 或使用原始长字符串，该特征既降低了基数，又保留了物理位置信号，通常与乘客等级、票价等高度互补，能提升模型表现。

## 适用场景
存在带有层级前缀或区段标识的字符串列（如机舱座位、酒店房间号、货架编号），且该标识与目标变量可能相关时使用。

## 代码模式
```python
import pandas as pd
import numpy as np

df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("U")
df["Deck"] = df["Deck"].astype("category")
```

## 注意事项
若字符串格式不统一（如甲板字母不在首字符），需要调整提取逻辑；缺失值填充为单独类别仅适用于缺失本身的机制可能与目标相关的情形，若完全随机缺失则可考虑其他策略；某些甲板类别可能仅出现在测试集，需提前对齐类别字典。
