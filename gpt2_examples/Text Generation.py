import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------Phase 1: Generate a token step by step (Using GPT2)--------------------
model_name = "/Applications/All/py code/pythonProject/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Print model information
print("Model Information:")
print(model)

prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")

# `inputs` is a dictionary: {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
# The shape of input_ids is (batch_size, sequence_length), where batch_size=1 and sequence_length=7 (number of tokens)
print()
print("Inputs:")
print(inputs)

with torch.no_grad():
    outputs = model(**inputs)

# The shape of logits is (batch_size, sequence_length, vocab_size),
# where vocab_size is the size of GPT-2's vocabulary (around 50,257)
logits = outputs.logits
print()
print(logits.shape)

# Here we extract the logits vector of the first sample (batch=0) at the last position (-1)
last_logits = logits[0, -1, :]
next_token_id = last_logits.argmax()  # ID of the most probable next token
print()
print(next_token_id)

next_word = tokenizer.decode(next_token_id)  # Decode token ID to text
print()
print(next_word)

# `torch.topk(last_logits, k=10)` returns a tuple (values, indices),
# where values are the top-k scores from last_logits, and indices are the corresponding token IDs
top_k = torch.topk(last_logits, k=10)
tokens = [tokenizer.decode(tk) for tk in top_k.indices]
related_values = [values for values in top_k.values]

# The `tokens` list stores the top 10 most likely next tokens (or token fragments) as text
print()
print(tokens)
print(related_values)


# --------------------Phase 2: Generate several tokens and show the time used--------------------
# Define token generation function
def generate_token(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id


generated_tokens = []
next_inputs = inputs
durations_s = []  # Time taken for each token generation

for _ in range(10):
    t0 = time.time()
    next_token_id = generate_token(next_inputs)
    durations_s += [time.time() - t0]

    # Concatenate the newly generated token to the existing input sequence
    next_inputs = {
        "input_ids": torch.cat(
            [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
            dim=1),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
    }

    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print()
print(f"{sum(durations_s)} s")  # Total time taken to generate the tokens above
print(generated_tokens)


# --------------------Phase 3: Speeding up text generation with KV-caching--------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values


generated_tokens = []
next_inputs = inputs
durations_cached_s = []

for _ in range(10):
    t0 = time.time()
    next_token_id, past_key_values = generate_token_with_past(next_inputs)
    durations_cached_s += [time.time() - t0]

    next_inputs = {
        "input_ids": next_token_id.reshape((1, 1)),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
        "past_key_values": past_key_values,
    }

    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print(f"{sum(durations_cached_s)} s")
print(generated_tokens)

# Visualize the time taken to generate each token
# We observe that the time per token generally increases as the sequence grows,
# but the first token takes longer as well, possibly due to cache warm-up.
# It's clear that using KV-caching significantly reduces generation time
plt.plot(durations_s, label="Without KV Cache")
plt.plot(durations_cached_s, label="With KV Cache")
plt.xlabel("Token Index")
plt.ylabel("Generation Time (s)")
plt.legend()
plt.title("Token Generation Time Comparison")
plt.show()
