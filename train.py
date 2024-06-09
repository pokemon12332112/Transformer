import torch.optim as optim
import math
import torch.nn as nn
from dataset import TranslationDataset
from model import TransformerModel
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = pq.read_table(r'D:\Transformer\train-00000-of-00001.parquet').to_pandas()
english_texts = [data[1]['en'] for data in np.array(dataset)]
hungarian_texts = [data[1]['hu'] for data in np.array(dataset)]

dataset = TranslationDataset(english_texts, hungarian_texts, tokenizer)
train_size = int(0.8*len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

vocab_size = tokenizer.vocab_size
model = TransformerModel(vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
            val_loss += loss.item()

    
    print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation loss: {val_loss/len(val_loader)}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'best_transformer_model_at_{epoch}.pth')