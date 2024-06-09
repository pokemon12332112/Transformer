import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from datasets import load_dataset
from transformers import BertTokenizer

class MultiHeadAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.depth = self.embed_dim // self.num_heads
        self.qkv = nn.Dense(self.embed_dim * 3, use_bias=False)
        self.proj = nn.Dense(self.embed_dim, use_bias=False)

    def __call__(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, self.depth)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, depth)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.depth)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)
        attn_weights = nn.softmax(attn_weights, axis=-1)
        
        attn_output = jnp.einsum('bhqk,bhvd->bhqd', attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        return self.proj(attn_output)

class FeedForward(nn.Module):
    embed_dim: int
    ff_dim: int
    drop_out: float = 0.1

    def setup(self):
        self.fc1 = nn.Dense(self.ff_dim)
        self.fc2 = nn.Dense(self.embed_dim)
        self.dropout = nn.Dropout(self.drop_out)

    def __call__(self, x, train: bool = True):
        x = nn.relu(self.fc1(x))
        x = self.dropout(x, deterministic=not train)
        return self.fc2(x)

class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    drop_out: float = 0.1

    def setup(self):
        self.mha = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.ffn = FeedForward(self.embed_dim, self.ff_dim, self.drop_out)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.drop_out)

    def __call__(self, x, mask=None, train: bool = True):
        attn_output = self.mha(x, mask)
        x = self.ln1(x + self.dropout(attn_output, deterministic=not train))
        ffn_output = self.ffn(x, train)
        return self.ln2(x + self.dropout(ffn_output, deterministic=not train))

class TransformerDecoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    drop_out: float = 0.1

    def setup(self):
        self.mha1 = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.mha2 = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.ffn = FeedForward(self.embed_dim, self.ff_dim, self.drop_out)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.drop_out)

    def __call__(self, x, enc_output, src_mask=None, tgt_mask=None, train: bool = True):
        attn1 = self.mha1(x, tgt_mask)
        x = self.ln1(x + self.dropout(attn1, deterministic=not train))
        attn2 = self.mha2(x, src_mask)
        x = self.ln2(x + self.dropout(attn2, deterministic=not train))
        ffn_output = self.ffn(x, train)
        return self.ln3(x + self.dropout(ffn_output, deterministic=not train))

class Transformer(nn.Module):
    src_vocab_size: int
    tgt_vocab_size: int
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    drop_out: float = 0.1

    def setup(self):
        self.encoder_embed = nn.Embed(self.src_vocab_size, self.embed_dim)
        self.decoder_embed = nn.Embed(self.tgt_vocab_size, self.embed_dim)
        self.encoder_layers = [TransformerEncoderLayer(self.embed_dim, self.num_heads, self.ff_dim, self.drop_out) for _ in range(self.num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(self.embed_dim, self.num_heads, self.ff_dim, self.drop_out) for _ in range(self.num_layers)]
        self.fc_out = nn.Dense(self.tgt_vocab_size)

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None, train: bool = True):
        src = self.encoder_embed(src)
        tgt = self.decoder_embed(tgt)
        
        for layer in self.encoder_layers:
            src = layer(src, src_mask, train)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask, train)
        
        return self.fc_out(tgt)
