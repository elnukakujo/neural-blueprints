"""BERT - Bidirectional Encoder Representations from Transformers"""
import torch
import torch.nn as nn
from .transformer_base import BaseTransformer, TransformerBlock, PositionalEncoding
import math
from typing import Optional


class BERT(BaseTransformer):
    """
    BERT model - Encoder-only transformer for pre-training and fine-tuning.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, dropout: float = 0.1,
                 max_len: int = 512, num_segments: int = 2):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension (768 for BERT-base, 1024 for BERT-large)
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            num_segments: Number of segment types (typically 2 for sentence pairs)
        """
        super().__init__(vocab_size, d_model, num_heads, num_layers, d_ff, 
                        dropout, max_len, encoder_only=True)
        
        # Segment embeddings for sentence pair tasks
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        
        # Task-specific heads
        self.mlm_head = nn.Linear(d_model, vocab_size)  # Masked Language Modeling
        self.nsp_head = nn.Linear(d_model, 2)  # Next Sentence Prediction
        self.pooler = nn.Linear(d_model, d_model)  # For classification tasks
        self.pooler_activation = nn.Tanh()
    
    def forward(self, input_ids: torch.Tensor, 
                segment_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            segment_ids: Segment IDs for sentence pairs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Dictionary with encoder output and pooled output
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.encoder_blocks:
            x = block(x, src_mask=attention_mask)
        
        # Pooled output (first token [CLS])
        pooled_output = self.pooler_activation(self.pooler(x[:, 0]))
        
        return {
            "last_hidden_state": x,
            "pooled_output": pooled_output
        }
    
    def get_mlm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get logits for masked language modeling"""
        return self.mlm_head(hidden_states)
    
    def get_nsp_logits(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """Get logits for next sentence prediction"""
        return self.nsp_head(pooled_output)