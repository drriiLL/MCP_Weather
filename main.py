import torch
import re

text = open(r'C:\Users\David\Downloads\merged_data_1.txt', 'r', encoding='utf-8').read()

# Оставляем только русские буквы, базовую латиницу, цифры и знаки препинания
text = re.sub(r'[^а-яёА-ЯЁa-zA-Z0-9 \n.,!?;:\-—«»"]', '', text)
text = text[:10_000_000]


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)

print(f'Размер данных: {data.shape}')
print(f'Словарь: {vocab_size} символов')


# 90% на обучение, 10% на проверку
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f'Обучающая выборка: {len(train_data)}')
print(f'Тестовая выборка: {len(val_data)}')

# Параметры модели
block_size = 256  # сколько символов модель видит за раз
batch_size = 64   # сколько примеров обрабатывается параллельно
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Устройство: {device}')

# Функция которая нарезает случайные куски текста для обучения
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch('train')
print(f'Форма батча x: {xb.shape}')
print(f'Форма батча y: {yb.shape}')


import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Создаём модель
n_embd = 384
n_heads = 6
n_layers = 6

model = GPT(vocab_size, n_embd, n_heads, n_layers, block_size).to(device)
print(f'Параметров в модели: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')


# Оптимизатор
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Функция оценки потерь
@torch.no_grad()
def estimate_loss(eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Цикл обучения
max_iters = 5000
eval_interval = 500

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Шаг {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Обучение завершено!')

# Генерация текста
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print('\nСгенерированный текст:')
print(generated)