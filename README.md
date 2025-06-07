# LLM-Inference

This repository contains code from the DeepLearning.AI course **Efficiently Serving LLMs**, along with my personal annotations and explanations. It serves as a comprehensive course notebook.

## Introduction

Inference optimization techniques for large language models (LLMs) have become a key research focus. As the parameter size of models continues to grow, the computational burden and memory requirements significantly increase, resulting in reduced inference efficiency. Therefore, improving inference speed is crucial.

Quantization techniques reduce memory consumption by lowering the numerical precision of model parameters, thereby accelerating inference. The primary goal is to achieve faster inference with minimal impact on model performance. Currently, quantization methods are mainly divided into two categories:

- **Quantization-Aware Training (QAT)**
- **Post-Training Quantization (PTQ)**, with PTQ being the primary focus of research.

## Details
Lesson 1-3 cover the fundamentals in detail, demonstrating the GPT-2 inference process and how batching accelerates inference. As shown in the figure below, when the batch size increases, throughput and latency both increases.
![Throughput and Latency](/Applications/All/py code/pythonProject/Efficiently Serving LLMs/img1.jpg)
