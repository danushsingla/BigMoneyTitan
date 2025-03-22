import os
import sys
import numpy as np

# Fixes the path from which this file is run
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import torch
from model import BigramLanguageModel
from tokenizers.tokenizer import RegexTokenizer

# Tries to run on GPU if it exists, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Uses basic RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.load("tokenizers/tokenizer_weights/regex_tokenizer.model")

# Initializing hyperparamaters for model (check model.py for each hyperparameter description)
batch_size = 4
block_size = 64 # Context token length
max_iters = 1000
eval_interval = 20
learning_rate = 1e-4
eval_iters = 1
n_embd = 384    # must be multiple of n_head
n_head = 24
n_layer = 6
dropout = 0.2

# Load the model
model = BigramLanguageModel(batch_size=batch_size, block_size=block_size, max_iters=max_iters, eval_interval=eval_interval, learning_rate=learning_rate, eval_iters=eval_iters, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout, tokenizer=tokenizer)
model = model.to(device)
model.load_state_dict(torch.load('weights/model.pth'))

# Start with some text, then have the model generate the rest
context = tokenizer.encode("Gandalf said to Frodo, ")
context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id)[0].tolist()))