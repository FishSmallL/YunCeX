---
name: gated-linear-unit-activation
description: 使用门控线性单元（GLU）作为神经网络层，通过元素级乘法门控机制动态过滤特征，提升模型对表格数据的非线性表达能力。
problems_solved: 高维特征交互学习、梯度流改进、特征自适应选择
keyword: machine learning
category: 模型架构
source_kernel: 1_ICR_adv_model
source_competition: icr-identify-age-related-conditions
created: 2026-05-11
---

# Gated Linear Unit Activation

## 可解决的问题
高维特征交互学习、梯度流改进、特征自适应选择

## 技巧说明
定义GatedLinearUnit层，包含两个并行的Dense子层：一个线性变换（无激活），另一个带sigmoid激活。输出为 linear(x) * sigmoid(x)。这样实现了对每个隐藏单元的门控——sigmoid输出0~1之间的门控系数，控制线性输出的信息流入下一层。门控机制允许网络跳过不相关信息，形成软特征选择；同时乘法交互引入了更强的非线性。相比普通Dense+ReLU，GLU能更灵活地控制信息流，且梯度传递更优（门控打开时近似线性）；相比标准GLU（通常用于卷积网络），此处直接应用于全连接层，适用于表格数据建模，可与GRN等结构结合提升性能。

## 适用场景
构建深度表格模型时，尤其在使用特征选择网络或门控残差网络时；输入维度较高且特征存在冗余，希望网络自动筛选有用特征。

## 代码模式
```python
import tensorflow as tf
from tensorflow.keras import layers as L

class GatedLinearUnit(L.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.linear = L.Dense(units)
        self.sigmoid = L.Dense(units, activation='sigmoid')
        self.units = units

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
```

## 注意事项
门控单元可能会引入额外参数量（双倍Dense）；sigmoid可能饱和导致梯度消失，可考虑使用门控线性单元变体（如GELU）；训练时可配合dropout与残差连接以获得更稳定效果。
