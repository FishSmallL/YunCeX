---
name: retrieval-ensemble-with-weighted-similarity
description: "使用MiniLM和BGE-small双塔模型编码问题，加权融合内积相似度进行段落检索，提升召回多样性与鲁棒性。"
use_case: 解决开放域问答中第一阶段段落检索的召回能力不足问题；适用需要融合多个语言模型嵌入的检索场景。
keyword: nlp
category: 集成与后处理
competition_type: 自然语言处理-多项选择问答
estimated_impact: 高
source_kernel: Podpall_2
source_competition: kaggle-llm-science-exam
created: 2026-05-14
---

# Retrieval Ensemble With Weighted Similarity

## 用途与场景
解决开放域问答中第一阶段段落检索的召回能力不足问题；适用需要融合多个语言模型嵌入的检索场景。

## 技巧说明
针对每个问题，同时用MiniLM和BGE-small两个双编码器分别对拼接了选项的prompt进行编码，得到384维向量；与预先存储的Wikipedia段落嵌入计算点积相似度，再按权重 MINILM_WEIGHT=0.5 进行线性融合：ensemble_sim = 0.5*minilm_sim + 0.5*bge_sim。融合后的相似度在给定段落分片内取top-k索引，所有分片合并后再统一排序得到最终候选段落。两模型在语义空间上互补，能捕获不同粒度的相关性，相比单一模型显著提升第一阶段检索召回率，同时避免了对单个模型过拟合的风险。

## 代码模式
```python
import torch
from sentence_transformers import SentenceTransformer

# 加载编码好的问题嵌入和段落嵌入
def calculate_similarities(model_name, passages_embs_name, device):
    with open(f'encoded_questions_{model_name}.pkl', 'rb') as f:
        query_embs = torch.tensor(pickle.load(f), dtype=torch.float16, device=device)
    passage_embs = torch.load(passages_embs_name, map_location=device)
    sims = torch.matmul(query_embs, passage_embs.T)
    return sims.cpu()

# 加权融合
minilm_sim = calculate_similarities('minilm', passage_file, 'cuda:0')
bge_sim = calculate_similarities('bge', passage_file, 'cuda:1')
ensemble_sim = 0.5 * minilm_sim + 0.5 * bge_sim
top_indices = torch.topk(ensemble_sim, k=500, dim=1).indices
```

## 注意事项
权重可通过验证集调整；两模型输出维度需相同；注意分批计算时两次编码并行在不同的GPU上运行，避免显存溢出。
