import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Fixes the path from which this file is run (runs from Tuto not from dataset directory)
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tiktoken
from tokenizer import Tokenizer, RegexTokenizer
from dataset.MATH import MATHDataset

num_merges=50

# Dataset hyperparameters
dataroot = "dataset/MATH/train/algebra/*"
max_tokens = 1024
mode = "train"
mode_answer = "full"
len_multiplier = 1.0
packing = False
randomize = True
pack_end = True
clean_numbers = False
latex_mask = False
peek_fraction=(0.1, 1.0)

# Create a new Tokenizer object
tokenizer = RegexTokenizer()

# # Register special tokens
# n = tokenizer.size()
# tokenizer.register_special_tokens({"<eos>": n, "<pad>":n+1})
# tokenizer.register_additional_tokens(["\\begin", "\\end", "\\boxed",
#                                    "\\cline", "\\rule", "\\sqrt", "\\frac", "\\left",
#                                    "\\qquad", "\\quad", "\\text", "\\underline",
#                                    "\\overline", "\\overrightarrow", "\\overleftarrow",
#                                    "\\overbrace", "\\underbrace", "\\hat", "\\bar",
#                                    "\\vec", "\\dot", "\\ddot", "\\dddot", "\\ddddot",
#                                    "\\tilde", "\\widetilde", "\\acute", "\\grave",
#                                    "\\check", "\\breve", "\\mathring", "\\matrix", "\\pmatrix"])

# Load natural language tokenizer
tokenizer.load("tokenizers/tokenizer_weights/regex_tokenizer_NT.model")


if __name__ == "__main__":
    # Initialize the dataset, reads in all samples within the dataroot directory regardless of max_tokens
    dataset = MATHDataset(dataroot, tokenizer, max_tokens, mode, mode_answer, len_multiplier, packing, randomize, pack_end, clean_numbers, latex_mask, peek_fraction)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    tokens = []

    # Train the tokenizer
    for i, batch in enumerate(dataloader):
        inputs = batch['input_ids']

        # Train only on inputs and remove all <pad> tokens
        token_list = inputs[0].tolist()
        if token_list[-1] == tokenizer.pad_token_id:
            token_list = token_list[:token_list.index(tokenizer.pad_token_id)]

        # Concatenate all tokens
        tokens.extend(token_list)

    # Decode and train on entire list
    text = tokenizer.decode(tokens)
    
    tokenizer.train(text, num_merges=100)

    tokenizer.save("tokenizers/tokenizer_weights/regex_tokenizer_MATH")