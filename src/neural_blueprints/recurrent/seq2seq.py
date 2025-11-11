"""Sequence-to-Sequence Models"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .rnn import LSTM
from ..base import NeuralNetworkBase


class EncoderDecoderNN(NeuralNetworkBase):
    """
    Encoder-Decoder architecture for sequence-to-sequence tasks.
    Base for machine translation, text generation, etc.
    """
    
    def __init__(self, input_vocab_size: int, output_vocab_size: int,
                 embedding_dim: int = 256, hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            input_vocab_size: Size of input vocabulary
            output_vocab_size: Size of output vocabulary
            embedding_dim: Dimension of embeddings
            hidden_size: Hidden dimension for RNN
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_rnn = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.decoder_rnn = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_projection = nn.Linear(hidden_size, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequence
        
        Args:
            src: Source sequence (batch, src_len)
            
        Returns:
            encoder_outputs: All encoder outputs
            hidden: Final encoder hidden state
        """
        embedded = self.dropout(self.encoder_embedding(src))
        encoder_outputs, hidden = self.encoder_rnn(embedded)
        return encoder_outputs, hidden
    
    def decode(self, tgt: torch.Tensor, 
               hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode target sequence
        
        Args:
            tgt: Target sequence (batch, tgt_len)
            hidden: Initial hidden state from encoder
            
        Returns:
            outputs: Decoder outputs
            hidden: Final decoder hidden state
        """
        embedded = self.dropout(self.decoder_embedding(tgt))
        decoder_outputs, hidden = self.decoder_rnn(embedded, hidden)
        outputs = self.output_projection(decoder_outputs)
        return outputs, hidden
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            
        Returns:
            outputs: Predicted logits (batch, tgt_len, vocab_size)
        """
        # Encode
        _, hidden = self.encode(src)
        
        # Decode
        outputs, _ = self.decode(tgt, hidden)
        
        return outputs
    
    def generate(self, src: torch.Tensor, max_length: int = 50,
                 start_token: int = 1, end_token: int = 2) -> torch.Tensor:
        """
        Generate sequence autoregressively
        
        Args:
            src: Source sequence
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            generated: Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encode
        _, hidden = self.encode(src)
        
        # Initialize with start token
        tgt = torch.full((batch_size, 1), start_token, 
                        dtype=torch.long, device=self.device)
        
        generated = []
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs, hidden = self.decode(tgt[:, -1:], hidden)
                next_token = outputs.argmax(dim=-1)
                generated.append(next_token)
                
                # Stop if all sequences have generated end token
                if (next_token == end_token).all():
                    break
                
                tgt = torch.cat([tgt, next_token], dim=1)
        
        return torch.cat(generated, dim=1)