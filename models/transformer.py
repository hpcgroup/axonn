import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None,:,:].expand(bsz, 1,-1)
        else:
            return pos_emb[None,:,:]

class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_length, dropout=0.1, eps=1e-12):
        super(GPTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class causallm_transformer(nn.Module):
    def __init__(self, n_layers, h_size, n_heads, vocab_size, max_seq_length, dropout=0.1):
        super(causallm_transformer, self).__init__()
        self.embeddings = GPTEmbeddings(vocab_size, h_size, max_seq_length)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(h_size, n_heads, 4*h_size, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(h_size)
        self.out_bias = nn.Parameter(torch.zeros((vocab_size,)))
    
    def forward(self, sentence, src_mask=None, src_key_padding_mask=None):
        out = self.embeddings(sentence)
        out = out.permute(1,0,2)
        for encoder_layer in self.encoder:
            out = encoder_layer(out, src_mask, src_key_padding_mask)
        out = self.layer_norm(out)
        out = out.permute(1,0,2).reshape(-1, out.shape[-1])
        out = F.linear(out, self.embeddings.word_embeddings.weight, self.out_bias)
        return out
