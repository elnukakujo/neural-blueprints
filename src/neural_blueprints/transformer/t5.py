"""T5 - Text-to-Text Transfer Transformer"""
import torch
import torch.nn as nn
from .transformer_base import BaseTransformer


class T5(BaseTransformer):
    """
    T5 model - Encoder-Decoder transformer for text-to-text tasks.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, max_len: int = 512):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        # Initialize with encoder layers first
        super().__init__(vocab_size, d_model, num_heads, num_encoder_layers, 
                        d_ff, dropout, max_len, encoder_only=False, decoder_only=False)
        
        # Override decoder blocks with correct number of layers
        from .transformer_base import TransformerBlock
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, is_decoder=True)
            for _ in range(num_decoder_layers)
        ])
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for encoder-decoder
        
        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            
        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        return super().forward(src, tgt)
    
    def generate(self, src: torch.Tensor, max_length: int = 50,
                 start_token_id: int = 0, end_token_id: int = 1) -> torch.Tensor:
        """
        Generate text autoregressively from source
        
        Args:
            src: Source sequence
            max_length: Maximum generation length
            start_token_id: ID of start token
            end_token_id: ID of end token
            
        Returns:
            Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encode source
        encoder_output = self.encode(src)
        
        # Initialize with start token
        tgt = torch.full((batch_size, 1), start_token_id, 
                        dtype=torch.long, device=self.device)
        
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask)
                logits = self.output_projection(decoder_output[:, -1:])
                next_token = logits.argmax(dim=-1)
                
                generated.append(next_token)
                
                if (next_token == end_token_id).all():
                    break
                
                tgt = torch.cat([tgt, next_token], dim=1)
        
        return torch.cat(generated, dim=1)