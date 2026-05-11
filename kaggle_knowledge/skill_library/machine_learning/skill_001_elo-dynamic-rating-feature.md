---
name: elo-dynamic-rating-feature
description: 为序列化对抗性事件（如体育比赛、游戏对局）中的参与者构建动态实时能力评分特征，捕捉随时间变化的实力消长，解决静态统计无法反映状态波动的问题。
problems_solved: "选手/队伍实时实力估计,序列比赛预测,冷启动能力评估,动态特征构建,历史信息压缩"
keyword: machine learning
category: 特征工程
source_kernel: 1_March 2026 sub
source_competition: march-machine-learning-mania-2026
created: 2026-05-11
---

# Elo Dynamic Rating Feature

## 可解决的问题
选手/队伍实时实力估计,序列比赛预测,冷启动能力评估,动态特征构建,历史信息压缩

## 技巧说明
利用 Elo 评级系统为每个参与者在每场比赛后更新其评分。具体做法：1) 按时间排序所有历史对局；2) 为每个队伍维护一个当前评分（初始值统一）；3) 对每场比赛，根据双方当前评分差通过逻辑函数计算预期胜率；4) 根据实际胜负结果（可用分差映射为胜率）与预期胜率的差值，按 K 因子调整双方评分。公式：新评分 = 旧评分 + K × (实际得分率 - 预期得分率)。实际得分率可用胜场设为1、负场设为0，或根据比分差映射到[0,1]。这样每场比赛后都更新评分，从而生成一条随比赛推移的动态能力曲线，可直接作为模型的数值特征。该方法的优势在于它模拟了贝叶斯更新过程：每次比赛带来的信息被压缩成一个连续的评分值，比简单地用历史胜率更能反映近期状态，并且自动处理了对手强弱的影响。相对于使用固定窗口的移动平均特征，Elo 能自适应地平衡近期与远期信息，K 值控制更新步长，宽参数可用于调整评分尺度的敏感度。此外，可扩展引入主场加成、评分下限等约束。

## 适用场景
数据集包含一系列对抗性事件（如体育比赛、电竞对局、棋类对局），且需要为每个参与者构建随事件顺序变化的能力特征，用于后续的序列预测模型（如LSTM、Transformer）或树模型。尤其适用于选手/队伍实力随时间变化明显的情况。

## 代码模式
```python
import numpy as np
import pandas as pd

def calculate_elo(teams, data, initial_rating=1500, k=40, width=400, lowerlim=-np.inf):
    '''
    teams : array-like, unique team ids
    data : pd.DataFrame, sorted chronologically with columns:
           'WTeamID', 'LTeamID', 'WScore', 'LScore', (optional) 'weight'
    returns : pd.DataFrame with original data and new columns 'WTeamElo', 'LTeamElo'
    '''
    if 'weight' not in data.columns:
        data['weight'] = 1.0
    elo = {team: initial_rating for team in teams}
    elo_w, elo_l = [], []
    for _, row in data.iterrows():
        w, l = row['WTeamID'], row['LTeamID']
        rw, rl = elo[w], elo[l]
        # Logistic expected score
        expected_w = 1.0 / (1.0 + 10 ** ((rl - rw) / width))
        expected_l = 1.0 - expected_w
        # Actual score based on point margin
        margin = row['WScore'] - row['LScore']
        actual_w = 1.0 if margin > 0 else 0.0  # simplify: win=1, loss=0
        # Update with weight and k
        k_w = k * row['weight']
        k_l = k * row['weight']
        new_rw = rw + k_w * (actual_w - expected_w)
        new_rl = rl + k_l * ((1 - actual_w) - expected_l)
        # Apply lower limit
        new_rw = max(new_rw, lowerlim)
        new_rl = max(new_rl, lowerlim)
        elo[w], elo[l] = new_rw, new_rl
        elo_w.append(new_rw)
        elo_l.append(new_rl)
    data['WTeamElo'] = elo_w
    data['LTeamElo'] = elo_l
    return data
```

## 注意事项
K值过大会导致评分波动过大，过小则对新信息不够敏感，通常可在20-100之间调整。宽度参数影响评分差与胜率的关系，可根据数据集的设计进行调整（如400对应约70%胜率对应200分差）。实际得分率除了硬编码胜负外，还可以用比分的逻辑函数映射到(0,1)，以减少信息损失。初始评分对早期对局影响较大，建议设定一个合理的初始值或使用预热期。注意数据必须严格按时间排序，否则会导致未来信息泄漏。如果数据量极小，评分可能不够稳定。
