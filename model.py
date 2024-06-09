import torch.nn as nn
from dataset import TranslationDataset
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_ffn=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.model_type = 'Transformer'
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, n_head, num_encoder_layers, num_decoder_layers, dim_ffn, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        # print(np.array(src).shape, np.array().shape)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

# vocab_size = tokenizer.vocab_size
# model = TransformerModel(vocab_size)
