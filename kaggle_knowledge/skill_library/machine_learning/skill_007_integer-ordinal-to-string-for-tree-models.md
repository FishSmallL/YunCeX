---
name: integer-ordinal-to-string-for-tree-models
description: 将表示类别的整数列显式转换为字符串再声明为类别类型，让梯度提升树模型将其视为无序类别而非连续数值，避免模型被迫学习虚假的单调关系。
problems_solved: 有序整数类别被模型误用作连续变量、树模型无法充分利用类别特征的非线性关系、数值编码引入不必要的顺序假设
keyword: machine learning
category: 数据预处理
source_kernel: 1_Titanic_CV_v2
source_competition: titanic
created: 2026-05-11
---

# Integer Ordinal To String For Tree Models

## 可解决的问题
有序整数类别被模型误用作连续变量、树模型无法充分利用类别特征的非线性关系、数值编码引入不必要的顺序假设

## 技巧说明
对于具有少量水平（如等级、地区编号）的整数列，先用 `.astype(str)` 转为字符串，再在 pandas 中设为 `category` 类型，或在使用 CatBoost/LightGBM 时指定 `cat_features` 索引。树模型本质是通过目标统计或分箱来处理类别特征，不依赖数值大小。将整数转为类别后，模型能够自由划分任意水平的组合，解除了对 1<2<3 顺序的依赖，从而捕捉到非单调关系，例如 Pclass=2 的生存率可能高于或低于 Pclass=1。相比直接使用整数并依赖模型的 ordinal 处理，这种显式类别声明更加稳定且可解释，在特征交叉和缺失值归因时也表现得更好。

## 适用场景
当整数列仅表示类别（有序或无序），且要输入 CatBoost、LightGBM 等支持原生类别特征的树模型时使用，尤其适用于水平数较少且与目标变量无严格单调关系的离散特征。

## 代码模式
```python
import pandas as pd

df["Pclass"] = df["Pclass"].astype(str)
cat_cols = ["Pclass", "Sex", "Embarked"]
for c in cat_cols:
    df[c] = df[c].astype("category")
```

## 注意事项
若使用不支持类别特征的模型需另作独热编码或序数编码；如果类别水平过多（如用户ID），转为类别可能导致内存膨胀，应考虑频率编码等替代方案；需确保训练集和测试集类别集合一致，否则推理时出现未知类会报错，可提前用 `pd.Categorical` 统一类别水平。
