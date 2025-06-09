# LLM-Inference
This repository presents key techniques for optimizing inference of large language models (LLMs). The code is adapted from the [*Efficiently Serving LLMs*](https://www.deeplearning.ai/short-courses/efficient-llm-inference/) by DeepLearning.AI and includes my personal annotations and explanations.
## Introduction

Inference optimization techniques for large language models (LLMs) have become a key research focus. As the parameter size of models continues to grow, the computational burden and memory requirements significantly increase, resulting in reduced inference efficiency. Therefore, improving inference speed is crucial.

Quantization techniques reduce memory consumption by lowering the numerical precision of model parameters, thereby accelerating inference. The primary goal is to achieve faster inference with minimal impact on model performance. Currently, quantization methods are mainly divided into two categories:

- **Quantization-Aware Training (QAT)**
- **Post-Training Quantization (PTQ)**, with PTQ being the primary focus of research.

## Details
### 1. GPT-2
The "gpt2_examples" directory covers the fundamentals in detail, demonstrating the GPT-2 inference process and how batching accelerates inference. As shown in the figure below, when the batch size increases, throughput and latency both increases.
<p align="center">
  <img src="imgs/img1.jpg" alt="Throughput and Latency" width="600"/>
</p>

### 2. Quantization
Quantization refers to the process of converting the weights and activations in a Large Language Model from high precision (such as FP32, i.e., 32-bit floating point) to lower precision (such as INT8 or INT4), in order to reduce model size, accelerate inference, and maintain accuracy as much as possible.
