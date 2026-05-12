---
name: multi-window-rolling-ewm-features
description: 对表格时序数据的每个特征列，并行生成多窗口滚动均值、滚动标准差与指数加权移动均值，高效扩充特征空间。
use_case: 适用于包含大量数值特征的时间序列表格数据，需要在少量延迟下快速生成趋势和波动性特征。尤其适合 Kaggle 时序预测或高频金融数据竞赛，模型类型可以是梯度提升树或线性模型。
keyword: Hull-Tactical-Market
category: 特征与表示
competition_type: 时序预测 | 表格回归
estimated_impact: 高
source_kernel: 1_Hull Tactical Prediction with CNN + Ensemble
source_competition: hull-tactical-market-prediction
created: 2026-05-12
---

# Multi Window Rolling Ewm Features

## 用途与场景
适用于包含大量数值特征的时间序列表格数据，需要在少量延迟下快速生成趋势和波动性特征。尤其适合 Kaggle 时序预测或高频金融数据竞赛，模型类型可以是梯度提升树或线性模型。

## 技巧说明
使用 Polars 的高性能表达式，对给定的基础特征列列表，一次性构建所有滚动/EWM 聚合表达式。具体做法：遍历每个特征列，针对预设的窗口大小列表（如 [5, 10, 20]）分别调用 rolling_mean、rolling_std 和 ewm_mean；将所有表达式收集后通过 with_columns 一次性应用到 DataFrame。这种方式避免了多次循环和中间表，利用 Polars 的惰性求值与并行优化，显著加快特征生成速度。相比使用 pandas 逐列计算，性能可提升 10 倍以上。

## 代码模式
```python
import polars as pl

def create_features(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    windows = [5, 10, 20]
    spans = [5, 10, 20]
    exprs = []
    for col_name in feature_cols:
        if col_name in df.columns:
            for w in windows:
                exprs.append(pl.col(col_name).rolling_mean(w).alias(f'{col_name}_roll_mean_{w}'))
                exprs.append(pl.col(col_name).rolling_std(w).alias(f'{col_name}_roll_std_{w}'))
            for s in spans:
                exprs.append(pl.col(col_name).ewm_mean(span=s).alias(f'{col_name}_ewm_mean_{s}'))
    return df.with_columns(exprs)
```

## 注意事项
窗口大小需根据数据的采样频率和预测目标调整；若数据存在缺失值，建议先填充以避免生成 null；超大窗口可能引入未来信息，需注意因果性。
