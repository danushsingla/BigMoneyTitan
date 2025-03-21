import torch
import torch.nn as nn
from model import BigramLanguageModel
from tokenizers.tokenizer import RegexTokenizer
from dataset.MATH import MATHDataset
from torch.utils.data import DataLoader, random_split

# Initializing hyperparameters for dataset (check testing_dataset for each hyperparameter description)
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
peek_fraction = (0.1, 1.0)

# Initializing hyperparamaters for model (check model.py for each hyperparameter description)
batch_size = 4
block_size = max_tokens # We want the context to be the entire question and answer
max_iters = 50
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
tokenizer.load("tokenizers/tokenizer_weights/regex_tokenizer_MATH.model")

# # Encode data
# with open('dataset/lotr.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

if __name__ == "__main__":
    # Initialize the dataset, reads in all samples within the dataroot directory regardless of max_tokens
    dataset = MATHDataset(dataroot, tokenizer, max_tokens, mode, mode_answer, len_multiplier, packing, randomize, pack_end, clean_numbers, latex_mask, peek_fraction)

    # Initialize the model and put it on the device
    model = BigramLanguageModel(batch_size=batch_size, block_size=block_size, max_iters=max_iters, eval_interval=eval_interval, learning_rate=learning_rate, eval_iters=eval_iters, n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout, tokenizer=tokenizer)
    model.to(device)
    # DELETE IF I DON'T WANT PRETRAINED
    model.load_state_dict(torch.load('weights/model_new.pth'))

    # create a PyTorch optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    # Train and test splits
    train_size = int(0.9*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def evaluate(model, eval_iters, device):
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            loss = 0
            for i, batch in enumerate(val_loader):
                X, Y = batch["input_ids"], batch["labels"]
                X, Y = X.to(device), Y.to(device)

                # Fix label by shifting back by one
                Y = Y[:, 1:]
                last_elements = torch.full((len(Y), 1), -100, dtype=Y.dtype, device=Y.device)
                Y = torch.cat((Y, last_elements), dim=1)

                logits, loss_temp = model(X, Y)
                loss += loss_temp
            loss /= len(val_loader)
            losses[k] = loss.item()
        out["val"] = losses.mean()
        model.train()
        return out

    for iter in range(max_iters):   
        # sample a batch of data -- the labels are incorrect, they must be shifted by one -- also why is eos token not at the end? is dataloader loading the data correctly?
        for i, batch in enumerate(train_loader):
            input, label = batch["input_ids"], batch["labels"]
            input, label = input.to(device), label.to(device)

            # Fix label by shifting back by one
            label = label[:, 1:]
            last_elements = torch.full((len(label), 1), -100, dtype=label.dtype, device=label.device)
            label = torch.cat((label, last_elements), dim=1)
            
            logits, loss = model(input, label)
            torch.set_printoptions(threshold=float('inf'))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # every once in a while evaluate the loss on train and val sets
            if i % eval_interval == 0:
                losses = evaluate(model, eval_iters, device)

                # Step the scheduler so that learning rate my change
                scheduler.step(losses['val'])

                print(f"iter {i} loss {loss.item():.4f} val loss {losses['val']:.4f}")
                torch.save(model.state_dict(), 'weights/model_new.pth')
                print("Model weights saved as", 'weights/model_new.pth')
            
            # After every 100 inputs, print some generated text
            if i % 100 == 0:
                print("Generated text: ")
                context = tokenizer.encode("Evaluate $a + b$ for $a = 5$ and $b = 3$.")
                context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                print(tokenizer.decode(model.generate(context, max_new_tokens=1000, eos_token_id=tokenizer.eos_token_id)[0].tolist()))

            # # DELETE
            # logits = logits[:, -1, :]
            # probs = F.softmax(logits, dim=-1)
            # idx_next = torch.multinomial(probs, num_samples=1)
            # break

    # save the model
    torch.save(model.state_dict(), 'weights/model_new.pth')
    print("Model weights saved as", 'weights/model_new.pth')

    # generate from the model
    context = torch.zeros(1, 1, dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=200)[0].tolist()))