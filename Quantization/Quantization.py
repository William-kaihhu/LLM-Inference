import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from utils import generate

model_name = '/Applications/All/py code/pythonProject/models/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"


def get_float32_dtype(self):
    return torch.float32


GPT2Model.dtype = property(get_float32_dtype)

print(model.get_memory_footprint())


# Quantize a tensor
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


t = model.transformer.h[0].attn.c_attn.weight.data
print(t)
print(t.shape)

t_q, state = quantize(t)
print()
print(t_q, t_q.min(), t_q.max())


# Dequantize a tensor
def dequantize(t, state):
    scale, zero_point = state
    return t.to(torch.float32) * scale + zero_point


t_rev = dequantize(t_q, state)
print(t_rev)
print(torch.abs(t - t_rev))  # Difference between original and dequantized data

# Generate output from the original model
response_expected = generate(
    model,
    tokenizer,
    [("The quick brown fox jumped over the", 10)]
)[0]
print(response_expected)


# Quantize the entire model
def quantize_model(model):
    states = {}
    for name, param in model.named_parameters():
        param.requires_grad = False
        param.data, state = quantize(param.data)
        states[name] = state
    return model, states


quant_model, states = quantize_model(model)
print(quant_model.get_memory_footprint())


# Calculate size of quantization states
def size_in_bytes(t):
    return t.numel() * t.element_size()


print(sum([
    size_in_bytes(v[0]) + size_in_bytes(v[1])
    for v in states.values()
]))


# Dequantize the model back to float
def dequantize_model(model, states):
    for name, param in model.named_parameters():
        state = states[name]
        param.data = dequantize(param.data, state)
    return model


dequant_model = dequantize_model(quant_model, states)
print(dequant_model.get_memory_footprint())

# Generate output from dequantized model
response_expected_2 = generate(
    dequant_model,
    tokenizer,
    [("The quick brown fox jumped over the", 10)]
)[0]
print(response_expected_2)
