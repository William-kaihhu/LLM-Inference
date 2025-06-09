import helpers
from helpers import init_batch, generate_next_token
from helpers import merge_batches, filter_batch, generate_batch_tokens_with_past, generate_batch
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '/Applications/All/py code/pythonProject/models/gpt2'
prompts = [
    "A wizard named Harry",
    "The rain in Spain falls",
    "What comes up must",
]
# seed the random number generator so our results are deterministic
random.seed(42)

# constants
queue_size = 32
batch_size = 8

# requests waiting to be processed
# this time requests are tuples (prompt, max_tokens)
request_queue = [
    (prompts[0], 100 if i % batch_size == 0 else 10)
    for i in range(queue_size)
]

t0 = time.time()
with tqdm(total=len(request_queue), desc=f"bs={batch_size}") as pbar:
    # first, let's seed the initial cached_batch
    # with the first `batch_size` inputs
    # and run the initial prefill step
    batch = init_batch(request_queue[:batch_size])
    cached_batch = generate_next_token(batch)
    request_queue = request_queue[batch_size:]

    # continue until both the request queue is
    # fully drained and every input
    # within the cached_batch has completed generation
    while (
            len(request_queue) > 0 or
            cached_batch["input_ids"].size(0) > 0
    ):
        batch_capacity = (
                batch_size - cached_batch["input_ids"].size(0)
        )
        if batch_capacity > 0 and len(request_queue) > 0:
            # prefill
            new_batch = init_batch(request_queue[:batch_capacity])
            new_batch = generate_next_token(new_batch)
            request_queue = request_queue[batch_capacity:]
            # merge
            cached_batch = merge_batches(cached_batch, new_batch)
        # decode
        cached_batch = generate_next_token(cached_batch)
        # remove any inputs that have finished generation
        cached_batch, removed_indices = filter_batch(cached_batch)
        pbar.update(len(removed_indices))

duration_s = time.time() - t0
print("duration", duration_s)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
# Padding on the left side to append tokens on the right side of the input
# Truncation: with "left" used to control cutting off from the left side
# when input exceeds the maximum length
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
generated_tokens = generate_batch(inputs, max_tokens=25)
for prompt, generated in zip(prompts, generated_tokens):
    print(prompt, f"\x1b[31m{generated}\x1b[0m\n")
