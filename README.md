# LLM-Inference
This repository presents key techniques for optimizing inference of large language models (LLMs). The code is adapted from the [*Efficiently Serving LLMs*](https://www.deeplearning.ai/short-courses/efficient-llm-inference/) by DeepLearning.AI and includes my personal annotations and explanations.
## Introduction

Inference optimization techniques for large language models (LLMs) have become a key research focus. As the parameter size of models continues to grow, the computational burden and memory requirements significantly increase, resulting in reduced inference efficiency. Therefore, it's crucial to improve inference speed.

Quantization techniques reduce memory consumption by lowering the numerical precision of model parameters, thereby accelerating inference. The primary goal is to achieve faster inference with minimal impact on model performance. Currently, quantization methods are mainly divided into two categories:

- **Quantization-Aware Training (QAT)**
- **Post-Training Quantization (PTQ)**, with PTQ being the primary focus of research.

## Details
### 1. GPT-2

The `gpt2_examples` folder provides a detailed demonstration of how the GPT-2 model performs text generation, along with optimization techniques to improve inference speed and efficiency.


#### Batching

**Batching** is the technique of processing multiple input samples in a single forward pass through the model. 
Instead of generating text for one input at a time, batching allows simultaneous computation for many inputs, 
which leads to:

- Better GPU/TPU utilization
- Reduced latency per sample
- Improved throughput in multi-user or production scenarios

#### KV Cache (Key-Value Cache)

**KV Cache** is an optimization used during auto-regressive text generation in transformer models.
Normally, the model recomputes attention over all previous tokens at every step. With KV caching:

- Keys and values from previous tokens are stored
- Only the new token is processed against cached keys/values
- Significantly faster generation, especially for long sequences

This caching mechanism enables efficient inference, particularly useful for chatbot, streaming, 
or server-based applications.
The efficiency improvement can be observed in Figure 1 and Figure 2.

<div align="center">
  <div style="display:inline-block; text-align:center; margin-right:40px;">
    <img src="imgs/img1.jpg" alt="Batching" width="500"/><br>
    <span style="font-size:12px">Figure 1: Batching</span>
  </div>
  <div style="display:inline-block; text-align:center;">
    <img src="imgs/img2.jpg" alt="With and without KV Cache" width="500"/><br>
    <span style="font-size:12px">Figure 2: With and without KV Cache</span>
  </div>
</div>





### 2. Quantization
Quantization refers to the process of converting the weights and activations in a Large Language Model from high precision (such as FP32, i.e., 32-bit floating point) to lower precision (such as INT8 or INT4).

The core logic of the quantization process is illustrated in the following code
```Python
def quantize(t):
    # obtain range of values in the tensor to map between 0 and 255
    min_val, max_val = t.min(), t.max()
    # determine the "zero-point", or value in the tensor to map to 0
    scale = (max_val - min_val) / 255
    zero_point = min_val
    # quantize and clamp to ensure we're in [0, 255]
    t_quant = (t - zero_point) / scale
    t_quant = torch.clamp(t_quant, min=0, max=255)
    # keep track of scale and zero_point for reversing quantization
    state = (scale, zero_point)
    # cast to uint8 and return
    t_quant = t_quant.type(torch.uint8)
    return t_quant, state
```
This function converts the original float32 tensor into an int8 representation, thereby greatly reducing memory consumption and improving inference speed.

However, quantization inevitably leads to some performance degradation. To illustrate this, we input the same prompt into both the original and quantized versions of GPT-2 and compare the generated outputs.

```python
response_expected = generate(
    model,
    tokenizer,
    [("The quick brown fox jumped over the", 10)]
)[0]
print(response_expected)
```
`output: The quick brown fox jumped over the fence and ran to the other side of the fence`
```python
response_expected_2 = generate(
    dequant_model,
    tokenizer,
    [("The quick brown fox jumped over the", 10)]
)[0]
print(response_expected_2)
```
`output: The quick brown fox jumped over the fence. The fox jumped over the fence`

It's obvious that even when the input is a very simple sentence, the model before quantization generates a better and more reasonable output.

Consequently, it's not sufficient to simply quantize the data without applying additional optimization techniques.

## Relative Research Insights

Quantization can be applied to weights, activations, and KV cache. 
Recent studies have uncovered several important findings.

**Key Matrix Channel Variance**  
   Some channels in the Key Matrix exhibit exceptionally large magnitudes. To avoid adversely affecting other channels, per-channel quantization is recommended ([*KIVI*](https://arxiv.org/abs/2402.02750)). 
Alternatively, these outlier channels can be decomposed and redistributed to other channels, balancing the activation magnitude distribution ([*QLLM*](https://arxiv.org/abs/2310.08041)).

**Middle-Deep Layer Similarity**  
   The middle and deep layers of large language models show high similarity. This enables techniques such as merging the middle-deep KV caches ([*Minicache*](https://arxiv.org/abs/2405.14366)) or directly pruning layers. The performance degradation caused by pruning can be mitigated through fine-tuning with QLoRA. 
However, improvements in question-answering tasks remain limited([*The Unreasonable Ineffectiveness of the Deeper Layers*](https://arxiv.org/abs/2403.17887)).

**Layer-Dependent Quantization Strategy**  
   Research indicates that shallow layers are critical for knowledge storage and information retrieval, while deeper layers contribute more to complex tasks such as mathematical reasoning. 
Therefore, quantization strategies should be adapted based on layer depth ([*CacheGen*](https://arxiv.org/abs/2310.07240)).

Inference optimization for large language models remains a critical and rapidly evolving area of research. 
As model sizes continue to grow exponentially, efficient deployment becomes increasingly challenging. 
Techniques such as quantization, pruning, and cache management play vital roles in reducing memory 
footprint and computational costs without significantly sacrificing model accuracy. Moreover, layer-wise 
and channel-wise adaptive strategies offer promising directions to balance performance and efficiency. 
Continued innovation in these methods will be essential to making large-scale models more accessible and 
practical for real-world applications.

This repository will be continuously updated with the latest papers and research advancements.