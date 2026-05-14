---
name: chunked-similarity-computation
description: 对4000个查询和超大段落库分批计算相似度矩阵，避免显存溢出，并最终合并排序得到全局top结果。
use_case: 解决大规模密集检索中全库相似度矩阵无法一次载入GPU内存的问题；适用于所有需要处理百万级以上文档嵌入的检索任务。
keyword: nlp
category: 数据处理
competition_type: 自然语言处理-多项选择问答
estimated_impact: 高
source_kernel: Podpall_2
source_competition: kaggle-llm-science-exam
created: 2026-05-14
---

# Chunked Similarity Computation

## 用途与场景
解决大规模密集检索中全库相似度矩阵无法一次载入GPU内存的问题；适用于所有需要处理百万级以上文档嵌入的检索任务。

## 技巧说明
整个Wikipedia索引包含约112个文件，每个文件存储100万条段落的嵌入（最后一个略少）。直接对全部段落同时计算会生成4000×112M的巨大矩阵，超过T4的14.8GB显存。做法是将段落文件逐个加载，形成4000×1M的相似度矩阵（float16仅占用7.45GB），每次计算后提取该分片的top-k索引及对应相似度，保存在pkl中。所有分片处理完毕后，将所有分片的top-k结果加载并拼接到一个数组中，再执行一次全局的argsort，选出最终的top段落。这样将空间复杂度从O(总段落数)降为O(分片大小+候选数)，能在单张T4上流畅运行，且支持多GPU并行计算不同模型的相似度。

## 代码模式
```python
import torch, pickle, os

def process_chunk(file_name):
    minilm_sim = calculate_similarities('minilm', file_name, 'cuda:0')
    bge_sim = calculate_similarities('bge', file_name, 'cuda:1')
    ensemble = 0.5*minilm_sim + 0.5*bge_sim
    top_vals, top_inds = torch.topk(ensemble, k=500, dim=1)
    with open(f'outs/scores_{file_name}', 'wb') as f:
        pickle.dump((top_vals, top_inds), f)

# 遍历所有分片
for fn in os.listdir('wikipedia_index'):
    process_chunk(fn)

# 合并分片
all_inds, all_scores = [], []
for fn in os.listdir('outs'):
    scores, inds = pickle.load(open(f'outs/{fn}', 'rb'))
    all_scores.append(scores); all_inds.append(inds)
final_scores = torch.cat(all_scores, dim=1)
final_inds = torch.cat(all_inds, dim=1)
top_global_inds = final_inds[torch.arange(len(final_inds)), final_scores.argsort(descending=True)[:, :500]]
```

## 注意事项
分片大小需根据显存调整，top-k值不宜过小以免丢失高相似段落；合并时注意对原始段落索引进行偏移校正（若分片索引未映射到全局索引）。
