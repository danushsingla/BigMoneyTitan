import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizer import Tokenizer, RegexTokenizer

dataset = "dataset/hobbit_intro.txt"

# Fixes any file path issues
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Tries to run on GPU if it exists, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a new Tokenizer object using Regex Tokenizer
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

if __name__ == "__main__":
    text = ""

    # Load the text from dataset file
    with open(dataset, 'r', encoding='utf-8') as f:
        text = f.read()

    # Train the tokenizer with the text
    tokenizer.train(text, num_merges=100)

    # Save the tokenizer
    tokenizer.save("tokenizers/tokenizer_weights/regex_tokenizer")