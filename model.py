import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers.tokenizer import BasicTokenizer, SimpleTokenizer, RegexTokenizer

# THIS IS A DECODER-ONLY TRANSFORMER MODEL, THUS MASKED MULTI-HEADED ATTENTION IS ALWAYS USED

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()

        # Initailize key, query, and value matrices
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Create a lower triangular matrix to mask out the lower half of the matrix
        # Ex. if block_size is 4, then the matrix will look like:
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        # Each sequence is row wise, so the first row is predicting the second token, second row is predicting the third token, etc.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Removes some parts of the input to prevent overfitting (reduce noise)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # B - batch size, T - sequence length, C - embedding size
        B, T, C = x.shape

        # We will use the analogy of searching the right book in a library when comparing key, value, and query

        # Apply linear transformations (weights) to the input for key (summaries of each book) and query (research question you look for in each book)
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Compute raw attention scores "affinities" and divide by dimensions to normalize (how much each book actually means to your research)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)

        # Masks out the lower half of the matrix, just like before, to prevent seeing attention scores of tokens that we have not yet gotten to
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T) mask out the lower half of the matrix

        # Converts each attention score into a value between 0 and 1, where all the scores then add to 1. This essentially
        # creates a matrix of floats that all sum to 1 (makes it easier when training than dealing with large numbers)
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # Drop some weights to prevent overfitting
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values (actual information from books used for your research)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        # Create multiple attention heads to work in parallel based on num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Concatenate the output of each head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # To further refine the results, combines the information from the concatenation
        # It's like soldering two pieces of metal, you can't just put them together and expect them to stick. The soldering iron is the projection
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        # Initialize a simple feed forward network with a non-linearity by going to a higher dimension, applying ReLU activation, then back
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        # Calculate head size (number of tokens per attention head)
        head_size = n_embd // n_head

        # Initialize the MultiHeadAttention layer to calculate the attention scores
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)

        # Initialize the FeedForward layer to calculate the final output
        self.ffwd = FeedForward(n_embd, dropout)

        # Initialize the LayerNorm layers to normalize the output
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        # Layer Norm is applied before, deviation from paper (thought to be better by Karpathy, but more testing needed)
        # Also, notice Add & Norm being used, where we add x and also normalize using l1 and l2
        # The reason for this addition is that, after softmax, weights can become so small they don't matter
        # so we add the original input to the output to prevent this from happening to extend its life a little
        x = x + self.sa(self.ln1(x))

        # Same thing but for feed forward. The point of feed forward is to introduce non-linearity by extending the data
        # to a higher dimension, allowing for it to understand more complex processes behind the code
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, batch_size=32, block_size=16, max_iters=15000, eval_interval=500, learning_rate=1e-3, eval_iters=200, 
                 n_embd=384, n_head=24, n_layer=6, dropout=0.2, tokenizer=None):
        super().__init__()

        # Initializing hyperparamaters
        self.batch_size = batch_size # how many independent sequences will we process in parallel?
        self.block_size = block_size # what is the maximum context length for predictions?
        self.max_iters = max_iters  # how many iterations will we train for?
        self.eval_interval = eval_interval  # how often will we evaluate our model's performance?
        self.learning_rate = learning_rate  # low learning rate since attention cannot tolerate high learning rates, and if you lower this then increase max_iters to compensate
        self.eval_iters = eval_iters # how many iterations will we use to evaluate our model's performance?
        self.n_embd = n_embd    # 384, has to be divisible by n_head (6)
        self.n_head = n_head    # number of heads (parallels) in the multiheadattention models
        self.n_layer = n_layer  # number of layers (blocks) in the transformer model
        self.dropout = dropout  # dropout rate
        self.tokenizer = tokenizer  # tokenizer object

        # Try to select CUDA device (GPU) otherwise CPU is fine but slower
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set seed for numpy and torch so that, when running, the results don't vary each time
        torch.manual_seed(1337)
        print("Currently on device: ", self.device)

        # The tokenizer acts like a personal dictionary for the transformer model, we can get the number of tokens from it
        self.vocab_size = self.tokenizer.size()

        # Each token directly reads off the logits for the next token from a lookup table (already created for us by some Python God)
        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embd)

        # Positional embeddings are added to the token embeddings to give the model some sense of order (improves performance)
        # This is due to the fact that the transformer model, since it parallelizes the heads, becomes unaware of the order of text
        # So we essentially give embeddings to each block as well so they can be identified later
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Create all the blocks -- divide the embeddings between the heads and go through feed forward
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.dropout, self.block_size) for _ in range(n_layer)])

        # Final layer norm to normalize the output
        self.ln_f = nn.LayerNorm(n_embd)

        # To go from token embeddings back to the true vocab size
        self.lm_head = nn.Linear(n_embd, self.vocab_size)
    
    def forward(self,idx, targets=None):
        # Each index is a batch of tokens (B,T) where B is the batch size (# sequences in batch) and T is the sequence length (# tokens in each sequence)
        B, T = idx.shape

        # Gets each token's embeddings from the embedding table -- idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # Generates a position embedding for each token in the sequence (takes a number in [0,1,2,3,4] and attributes a vector to it)
        # Since T is the number of tokens in a sequence, this generates a position embedding for each token in the sequence
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C); torch.arange(5) generates [0,1,2,3,4]

        # Add the token and position embeddings together to get the final embeddings -- (B,T,C) (Python scales up the pos_emb tensor to match the tok_emb tensor)
        x = tok_emb + pos_emb # (B,T,C)

        # Push code through the blocks (getting attention scores, training)
        x = self.blocks(x) # (B,T,C)

        # Normalize the output
        x = self.ln_f(x) # (B,T,C)

        # Get the logits (probability predictions) for the next token. Attributes a percentage likliehood for the next token based on each token from the dictionary
        logits = self.lm_head(x) # (B,T, vocab_size)

        # Calculate loss (optional) if targets are present
        if targets is None:
            loss = None
        else:
            # Get the shape of the logits tensor
            B, T, C = logits.shape

            # Due to requirements from cross entropy, modify the shapes of logits and targets to pass it through and generate loss
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    # eos_token_id is the token that signifies the end of a sequence, it can hold a special ID in the token dictionary that we would need to specify
    # (otherwise the model won't ever know when to stop)
    def generate(self, idx, max_new_tokens, eos_token_id=None):
        # idx is (B,T) array of indices in the current context
        # For each new token we want to generate, up to max_new_tokens
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens (if block size is 8, then only obtain the last 8 tokens of the sequence for attention)
            # We do this so that the model does not confuse itself with too many tokens at once
            idx_cond = idx[:, -self.block_size:]

            # Put this block through forward, where it will use its trained weights to get the predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step (only look at prob distribution for last tokens)
            logits = logits[:, -1, :] # becomes (B, C)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)

            # Sample from the distribution rather than taking most likely token (introduces diversity)
            # This ensures the text sounds less like a robot and more like a human
            # It also allows the model to explore different paths and not get stuck in a loop
            # There is a chance, however, of it picking a low-probability token but it's worth it
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

            # Look for eos token and stop there
            if eos_token_id is not None:
                if int(idx_next.item()) == eos_token_id:
                    return idx
            
            # Append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx