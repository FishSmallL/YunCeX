---
name: elo-dynamic-rating
description: 利用Elo算法从序列对战中生成随时间变化的动态实力评分，解决体育/游戏预测中队伍（或选手）实力动态建模问题，输出可直接作为模型特征的连续数值，并适应实力概念漂移。
problems_solved: 时间序列实力评估、动态排名特征生成、冷启动评分、序列对战中的胜率概率估计、概念漂移下的特征自适应
keyword: machine learning
category: 特征工程
source_kernel: 1_March 2026 sub
source_competition: march-machine-learning-mania-2026
created: 2026-05-11
---

# Elo Dynamic Rating

## 可解决的问题
时间序列实力评估、动态排名特征生成、冷启动评分、序列对战中的胜率概率估计、概念漂移下的特征自适应

## 技巧说明
按比赛时间顺序迭代每条记录，维护各队伍的当前评分。每场比赛预测胜率为 E = 1/(1+10^((对手评分-当前评分)/宽度))，实际结果 S（赢=1，输=0），评分更新为 R_new = R_old + K*(S - E)。K 控制每次更新的幅度，宽度控制评分差对预期胜率的敏感度。通过将最新评分或赛前评分作为特征，模型可直接利用历史对战信息。该方法的优势在于：1) 动态性：近期表现影响更大，自动衰减远期信息；2) 平滑性：相比滚动胜率等统计量，Elo 避免窗口选择的随意性和由于比赛场次不均导致的方差问题；3) 可解释性：评分差与预期胜率具有单调关系。同时支持对比赛重要性加权（如锦标赛高于友谊赛），只需在更新时将 K 乘以上述权重即可。

## 适用场景
存在按时间排序的对抗性比赛记录（双方明确、胜负已知），需要为参赛实体生成动态特征，用于后续胜负预测、让分预测或排名任务。模型可以是逻辑回归、树模型、神经网络等。

## 代码模式
```python
import numpy as np
import pandas as pd

def calculate_elo(teams, data, initial_rating=1500, k=32, width=400):
    ratings = {team: initial_rating for team in teams}
    for _, row in data.iterrows():
        t1, t2 = row['team1'], row['team2']
        r1, r2 = ratings[t1], ratings[t2]
        e1 = 1.0 / (1 + 10 ** ((r2 - r1) / width))
        e2 = 1.0 - e1
        s1 = row['result']
        s2 = 1 - s1
        ratings[t1] = r1 + k * (s1 - e1)
        ratings[t2] = r2 + k * (s2 - e2)
    return ratings
```

## 注意事项
K 值选取影响收敛速度与评分波动，通常通过验证集决定，32~140 是常见范围。初始评分不敏感但需注意冷启动：早期比赛可能使评分不准确，可考虑在数据充足时仅使用后期评分。若比赛间隔差异大，可结合时间衰减机制。宽度通常设为 400（与排名系统的逻辑分布对应），但可通过交叉验证调整。
