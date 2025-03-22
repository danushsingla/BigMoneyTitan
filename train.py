import torch
import torch.nn as nn
from model import BigramLanguageModel
from tokenizers.tokenizer import RegexTokenizer
from torch.utils.data import DataLoader, random_split

dataset = "dataset/hobbit_intro.txt"

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = RegexTokenizer()
# Loads the tokenizer that has already been trained
tokenizer.load("tokenizers/tokenizer_weights/regex_tokenizer.model")

# Loss function to calculate loss, use no grad so this doesn't affect training weights
@torch.no_grad()
def estimate_loss():
    # The out is a dictionary, where the key is the split and the value is the loss
    out = {}

    # Set the model to evaluation mode (doesn't do anything since I never set it)
    model.eval()

    # Calculates loss for both train and val splits
    for split in ['train', 'val']:
        # Initialize the loss tensor with how many iterations of evaluation it has to do
        losses = torch.zeros(eval_iters)

        # For each iteration, get a batch of data and calculate the loss
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        # Store the average loss for the split
        out[split] = losses.mean()

    # Set the model back to training mode
    model.train()
    return out

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])

    # Notice how y is always one ahead of x, so that the model can learn to predict the next token
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Encode data
with open(dataset, 'r', encoding='utf-8') as f:
    text = f.read()
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Initialize the model and put it on the device
model = BigramLanguageModel(batch_size=batch_size, block_size=block_size, max_iters=max_iters, eval_interval=eval_interval, learning_rate=learning_rate, eval_iters=eval_iters, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout, tokenizer=tokenizer)
model.to(device)

# # DELETE IF I DON'T WANT PRETRAINED
# model.load_state_dict(torch.load('weights/model_new.pth'))

# Create a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets and print them
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)

    # Update the weights (actual training)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'weights/model.pth')
print("Model weights saved as", 'weights/model.pth')