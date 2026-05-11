---
name: family-size-and-isalone-composite-features
description: 将兄弟姐妹/配偶数、父母/子女数相加得到家庭总人数，并衍生二值特征标记是否独行，从多列关系中提取群体效应信号。
problems_solved: 相关特征未进行组合导致信息冗余、家庭结构信息利用率低、群体互动效应捕捉
keyword: machine learning
category: 特征工程
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Family Size And Isalone Composite Features

## 可解决的问题
相关特征未进行组合导致信息冗余、家庭结构信息利用率低、群体互动效应捕捉

## 技巧说明
将 `SibSp` 和 `Parch` 相加并加 1（自身），得到 `FamilySize` 表示同行家庭总人数；进一步构建布尔特征 `IsAlone`（FamilySize == 1）标记独自出行。这种组合能直接反映群体规模对生存率的影响，例如大群体可能互相协助或延误撤离。IsAlone 特征则将独行状态清晰独立出来，让模型更容易学习到独行与结伴之间的非对称效应。相比直接分别使用原始两列，组合特征降低了模型推断交互效应的难度，并在树模型中减少了分裂次数；二值指示符尤其擅长捕捉阈值效应。

## 适用场景
数据集包含多个衡量群体人数或成员关系的数值列，且目标变量可能受群体规模非线性影响时使用，适用于分类、回归及生存分析。

## 代码模式
```python
import pandas as pd

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

## 注意事项
如果源列存在缺失值需先填补；FamilySize 可能呈现 U 型或倒 U 型模式，可进一步分箱或与年龄等交互；IsAlone 仅在家庭定义准确时有效，需确保同组旅客与 FamilySize 口径一致。
