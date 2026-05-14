---
name: cross-encoder-reranking
description: 对双塔检索得到的候选段落用交叉编码器精排，选取最相关的少量段落构建最终上下文，大幅提升精排准确性。
use_case: 解决粗排召回结果中噪声多、相关性不精确的问题；适用于需要高精度排序的问答系统或信息检索精排阶段。
keyword: nlp
category: 模型设计
competition_type: 自然语言处理-多项选择问答
estimated_impact: 高
source_kernel: Podpall_2
source_competition: kaggle-llm-science-exam
created: 2026-05-14
---

# Cross Encoder Reranking

## 用途与场景
解决粗排召回结果中噪声多、相关性不精确的问题；适用于需要高精度排序的问答系统或信息检索精排阶段。

## 技巧说明
第一阶段双塔检索返回top-500段落，但由于双塔模型将问题和段落独立编码，交互有限，粗排结果仍有噪声。第二阶段引入交叉编码器（如DeBERTa或Electra），将(问题, 段落)成对输入，通过全自注意力机制充分融合二者信息，输出相关性分数。对每个问题的500个候选逐一打分，最后按分数排序取前10个段落拼接作为最终阅读理解的上下文。这一重排序流程兼顾了效率（只在少量候选上运行重型模型）与效果（交叉编码器的深度交互显著提升排序精度），比单纯依赖双塔或传统词袋重排序方法更能保证最终答案的证据质量。

## 代码模式
```python
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def do_cross_encoder_scoring(model, tokenizer, pairs):
    dataloader = DataLoader(
        pairs, batch_size=32,
        collate_fn=lambda batch: tokenizer(
            batch, padding=True, truncation=True,
            return_tensors='pt', max_length=512)
    )
    scores = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**batch).logits
            scores.extend(logits[:, 1].cpu().numpy())  # 正类得分
    return np.array(scores)

# 使用
reranker_model = AutoModelForSequenceClassification.from_pretrained('path').cuda()
tokenizer = AutoTokenizer.from_pretrained('path')
question = "..."
candidates = ["passage1", "passage2", ...]
pairs = [(question, p) for p in candidates]
scores = do_cross_encoder_scoring(reranker_model, tokenizer, pairs)
top10_indices = np.argsort(scores)[-10:][::-1]
```

## 注意事项
候选数不宜过大（一般<1000），否则计算开销过大；输入截断长度需权衡覆盖与效率；可以使用单塔双向注意力模型作为重排序器。
