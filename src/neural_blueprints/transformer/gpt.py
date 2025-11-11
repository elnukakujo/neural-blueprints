"""GPT - Generative Pre-trained Transformer"""
import torch
from typing import Optional
from .transformer_base import BaseTransformer


class GPT(BaseTransformer):
    """
    GPT model - Decoder-only transformer for autoregressive generation.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, dropout: float = 0.1,
                 max_len: int = 1024):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__(vocab_size, d_model, num_heads, num_layers, d_ff, 
                        dropout, max_len, decoder_only=True)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        return super().forward(input_ids)
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            prompt: Initial prompt tokens (batch, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated sequence (batch, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = prompt
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated