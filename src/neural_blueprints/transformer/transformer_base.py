"""Base Transformer Components"""
import torch
import torch.nn as nn
import math
from typing import Optional
from ..base import NeuralNetworkBase


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer encoder/decoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, is_decoder: bool = False):
        super().__init__()
        
        self.is_decoder = is_decoder
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (only for decoder)
        if is_decoder:
            self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_output is not None:
            attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class BaseTransformer(NeuralNetworkBase):
    """
    Base Transformer architecture.
    Can be configured as encoder-only, decoder-only, or encoder-decoder.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1,
                 max_len: int = 5000, encoder_only: bool = False, 
                 decoder_only: bool = False):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            encoder_only: Use only encoder (like BERT)
            decoder_only: Use only decoder (like GPT)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_only = encoder_only
        self.decoder_only = decoder_only
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        if not decoder_only:
            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder=False)
                for _ in range(num_layers)
            ])
        
        # Decoder
        if not encoder_only:
            self.decoder_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder=True)
                for _ in range(num_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.device)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence"""
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.encoder_blocks:
            x = block(x, src_mask=src_mask)
        
        return x
    
    def decode(self, tgt: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None, 
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence"""
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for block in self.decoder_blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        if self.encoder_only:
            # Encoder-only (BERT-like)
            return self.output_projection(self.encode(src))
        
        elif self.decoder_only:
            # Decoder-only (GPT-like)
            tgt_mask = self.generate_square_subsequent_mask(src.size(1))
            x = self.decode(src, tgt_mask=tgt_mask)
            return self.output_projection(x)
        
        else:
            # Encoder-decoder
            assert tgt is not None, "Target required for encoder-decoder"
            encoder_output = self.encode(src)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask)
            return self.output_projection(decoder_output)