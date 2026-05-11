---
name: elo-dynamic-rating-feature
description: 为具有历史对抗序列的结构化数据构建动态实力评分特征，解决需根据过往交战结果评估当前相对强度并生成时间窗特征的问题。
problems_solved: 时间序列评分、动态实力评估、历史对战序列特征构建、球员/团队实力预测、排序学习特征生成
keyword: machine learning
category: 特征工程
source_kernel: 1_March 2026 sub
source_competition: march-machine-learning-mania-2026
created: 2026-05-11
---

# Elo Dynamic Rating Feature

## 可解决的问题
时间序列评分、动态实力评估、历史对战序列特征构建、球员/团队实力预测、排序学习特征生成

## 技巧说明
基于Elo评分系统，为每个实体（如球队）维护一个随时间更新的评分。按时间顺序遍历每场比赛，根据双方当前评分计算期望胜率（使用逻辑函数，宽度参数控制分数差敏感度），再结合实际胜负结果更新评分：新评分 = 旧评分 + K * (实际结果 - 期望胜率)。K值可固定或随比赛重要性动态调整。该技术有效因为它将复杂的历史对战信息压缩为一个连续值，同时天然处理了实力随时间漂移的问题，并可通过调节K值平衡稳定性和适应性。相比于简单的胜率特征，Elo评分能反映对手强度和近期状态；相比于固定时间窗口统计，它无需存储完整历史，计算效率高且不易受早期数据主导。实现时支持加权版本（如使用比赛权重）、自定义初始评分、分数限制下限等，可扩展处理平局或比分差距。

## 适用场景
适用于任何具有成对比较结果且存在时序顺序的数据，如体育比赛、游戏对战、A/B测试变体比较或随时间变化的实体评估。需按时间排序的胜负记录（含双方ID），模型类型通常为梯度提升树或逻辑回归等需要数值实力特征的模型。

## 代码模式
```python
import numpy as np
import pandas as pd

def calculate_elo(teams, data, initial_rating=2000, k=140, width=400, weights=False, lowerlim=-np.inf):
    elo = {team: initial_rating for team in teams}
    ratings_history = []

    for _, row in data.iterrows():
        wteam, lteam = row['WTeamID'], row['LTeamID']
        rating_w, rating_l = elo[wteam], elo[lteam]

        expected_w = 1.0 / (1 + 10 ** ((rating_l - rating_w) / width))
        result_w = 1

        if weights and 'weight' in row:
            k_val = k * row['weight']
        else:
            k_val = k

        new_rating_w = rating_w + k_val * (result_w - expected_w)
        new_rating_l = rating_l + k_val * (0 - (1 - expected_w))

        new_rating_w = max(new_rating_w, lowerlim)
        new_rating_l = max(new_rating_l, lowerlim)

        elo[wteam] = new_rating_w
        elo[lteam] = new_rating_l

        ratings_history.append((wteam, new_rating_w))
        ratings_history.append((lteam, new_rating_l))

    return pd.DataFrame(ratings_history, columns=['TeamID', 'EloRating']).groupby('TeamID').last().reset_index()
```

## 注意事项
初始评分和K值对收敛速度和稳定性影响大，建议通过时间序列交叉验证调优。如果比赛数量极少或分布极不均匀，Elo可能不稳定，可添加回归均值机制。常见错误：忽略数据的时间顺序导致未来信息泄露；未根据预测目标调整评分敏感性（width参数）。
