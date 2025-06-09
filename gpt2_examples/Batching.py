import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "/Applications/All/py code/pythonProject/models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")


# --------------------Phase 1: Using KV Cache to generate texts from Lesson 1--------------------
def generate_token_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values


def generate(inputs, max_tokens):
    generated_tokens = []
    next_inputs = inputs
    for _ in range(max_tokens):
        next_token_id, past_key_values = generate_token_with_past(next_inputs)
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim=1
            ),
            "past_key_values": past_key_values,
        }
        next_token = tokenizer.decode(next_token_id)
        generated_tokens.append(next_token)
    return "".join(generated_tokens)


tokens = generate(inputs, max_tokens=10)
print(prompt + tokens)
print()

# --------------------Phase 2: Batching Process --------------------
# Define PAD Token = EOS Token = 50256. EOS stands for End Of Sequence.
# model.config is a configuration object in Hugging Face, containing various model settings
# (such as vocab size, hidden size, pad token ID, eos token ID, etc.)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
# Padding on the left so that new tokens are appended on the right
# truncation: truncate inputs longer than max length from the left
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Multiple input prompts
prompts = [
    "The quick brown fox jumped over the",
    "The rain in Spain falls",
    "What comes up must",
]
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
print("input_ids:", inputs["input_ids"])
print("shape:", inputs["input_ids"].shape)
# attention_mask=1 indicates tokens that should be attended to by the model. Padding tokens should not be attended to.
print("attention_mask:", inputs["attention_mask"])
print("shape:", inputs["attention_mask"].shape)

attention_mask = inputs["attention_mask"]
# cumsum(-1): cumulative sum along the last dimension,
# resulting in the number of valid (non-padding) tokens before each position
position_ids = attention_mask.long().cumsum(-1) - 1  # -1 → real token positions start from 0
position_ids.masked_fill_(attention_mask == 0, 1)  # Set padding token positions to 1 to avoid using index -1
print()
print(position_ids)

with torch.no_grad():
    outputs = model(position_ids=position_ids, **inputs)
logits = outputs.logits
last_logits = logits[:, -1, :]  # Select all batches’ final logits
next_token_ids = last_logits.argmax(dim=1)
print()
print(next_token_ids)
next_tokens = tokenizer.batch_decode(next_token_ids)
print(next_tokens)


# --------------------Phase 3: Generate all tokens for some max tokens--------------------
def generate_batch_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim=1)
    return next_token_ids, outputs.past_key_values


def generate_batch(inputs, max_tokens):
    # Create a list of generated tokens for each input in the batch
    generated_tokens = [[] for _ in range(inputs["input_ids"].shape[0])]

    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    next_inputs = {
        "position_ids": position_ids,
        **inputs
    }

    for _ in range(max_tokens):
        next_token_ids, past_key_values = generate_batch_tokens_with_past(next_inputs)

        # input_ids: (batch_size, 1) tensor of the newly generated tokens for this step
        # position_ids: take the last position index of the previous step,
        # add 1 to generate new position indices. Shape: (batch_size, 1)
        # attention_mask: append a column of 1s to the previous mask →
        # shape changes from (batch_size, t) → (batch_size, t+1), keeping alignment
        next_inputs = {
            "input_ids": next_token_ids.reshape((-1, 1)),
            "position_ids": next_inputs["position_ids"][:, -1].unsqueeze(-1) + 1,
            "attention_mask": torch.cat([
                next_inputs["attention_mask"],
                torch.ones((next_token_ids.shape[0], 1)),
            ], dim=1),
            "past_key_values": past_key_values,
        }

        next_tokens = tokenizer.batch_decode(next_token_ids)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
    return ["".join(tokens) for tokens in generated_tokens]


generated_tokens = generate_batch(inputs, max_tokens=20)
for prompt, generated in zip(prompts, generated_tokens):
    print(prompt, f"\x1b[31m{generated}\x1b[0m\n")

# --------------------Phase 4: Throughput and Latency--------------------

# Constants
max_tokens = 10

# Observations
durations = []
throughputs = []
latencies = []

batch_sizes = [2 ** p for p in range(8)]
for batch_size in batch_sizes:
    print(f"bs= {batch_size}")

    # Generate tokens for batch and record duration
    t0 = time.time()
    batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]
    inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt")
    generated_tokens = generate_batch(inputs, max_tokens=max_tokens)
    duration_s = time.time() - t0

    ntokens = batch_size * max_tokens
    throughput = ntokens / duration_s
    avg_latency = duration_s / max_tokens
    print("duration", duration_s)
    print("throughput", throughput)
    print("avg latency", avg_latency)
    print()

    durations.append(duration_s)
    throughputs.append(throughput)
    latencies.append(avg_latency)


def render_plot(x, y1, y2, x_label, y1_label, y2_label):
    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot the first line (throughput)
    color = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Set the x-axis to be log-scaled
    ax1.set_xscale('log', base=2)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(y2_label, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.show()


render_plot(
    batch_sizes,
    throughputs,
    latencies,
    "Batch Size",
    "Throughput",
    "Latency"
)
