# 🚀 nano-patch-sequence-pack

Just a few lines to combine 🤗 **Transformers**, **Flash Attention 2**, and `torch.compile` — simple, clean, fast ⚡

---

All of them now support **sequence packing** — removing unnecessary padding by packing a batch of tokens into one contiguous sequence.
But few frameworks make it easy to actually use.

**Not anymore!** With a tiny `patch.py`, you can enable packing and compiler-level optimization in just **two lines** — and seamlessly plug it into **any framework built on Transformers** 💡

```python
from patch import patch
patch(model)  # model loaded from Hugging Face Transformers
```

The logic inside `patch.py` is **clean and compact** — perfect for quick customization and extension.

Example from [Enhancing SFT Training Efficiency Using Packing and FlashAttention2 with Position IDs](https://github.com/huggingface/transformers/pull/31629)


### 📊 Example Result 1

**Dataset**: OrcaMath subset

**Setup**: FSDP with 8 GPUs

| Model      | Data Process | Time (s) | Throughput (token/s) | Memory (MB) |
| ---------- | ------------ | -------- | -------------------- | ----------- |
| Llama2-7B  | Padding      | 790      | 1269                 | 22305       |
| Llama2-7B  | ThisPR       | 574      | 1746                 | 20950       |
| Mistral-7B | Padding      | 812      | 1216                 | 23603       |
| Mistral-7B | ThisPR       | 596      | 1658                 | 22409       |


### 📊 Example Result 2

**Dataset**: FLAN subset

**Setup**: FSDP with 8 GPUs

| Model      | Data Process | Time (s) | Throughput (token/s) | Memory (MB) |
| ---------- | ------------ | -------- | -------------------- | ----------- |
| Llama2-7B  | Padding      | 1526     | 771                  | 29234       |
| Llama2-7B  | ThisPR       | 809      | 1455                 | 23854       |
| Mistral-7B | Padding      | 742      | 742                  | 30625       |
| Mistral-7B | ThisPR       | 1408     | 1408                 | 24549       |
