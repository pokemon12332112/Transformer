from jax_transformer import *
from jax_utils import *
from jax_dataset import *
import jax
import optax
from flax.training import train_state
from flax import linen as nn
import numpy as np
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import pyarrow.parquet as pq

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset("Helsinki-NLP/opus_books", "en-hu")
# english_texts = [data[1]['en'] for data in np.array(dataset)]
# hungarian_texts = [data[1]['hu'] for data in np.array(dataset)]
translation_dataset = TranslationDataset(dataset['train'], tokenizer, 'en', 'hu')

train_size = int(0.8 * len(translation_dataset))
val_size = int(0.1 * len(translation_dataset))
test_size = len(translation_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(translation_dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

learning_rate = 1e-4
rng = jax.random.PRNGKey(0)
model = Transformer(src_vocab_size=len(tokenizer), tgt_vocab_size=len(tokenizer))

state = create_train_state(rng, learning_rate, model, len(tokenizer), len(tokenizer))

for epoch in range(10):
    rng, input_rng = jax.random.split(rng)
    state, train_metrics = train_epoch(state, train_loader, input_rng)
    val_metrics = evaluate(state, val_loader)
    
    print(f"Epoch {epoch + 1}, Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, "
          f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

