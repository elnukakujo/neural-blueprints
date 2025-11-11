"""Example: Training a Transformer for sequence modeling"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from neural_blueprints.transformer import GPT


def create_dummy_data(vocab_size=1000, seq_len=50, num_samples=10000):
    """Create dummy sequence data"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    vocab_size = 1000
    seq_len = 50
    batch_size = 32
    
    # Create dummy data
    train_data = create_dummy_data(vocab_size, seq_len, 10000)
    train_dataset = TensorDataset(train_data[:-1].transpose(0, 1), 
                                   train_data[1:].transpose(0, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create GPT model
    model = GPT(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
        max_len=seq_len
    )
    model.to(device)
    
    print(f"Model has {model.count_parameters():,} parameters")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape for loss computation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Generate text
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
    generated = model.generate(prompt, max_new_tokens=40, temperature=0.8)
    print(f"\nGenerated sequence shape: {generated.shape}")
    
    # Save model
    model.save("gpt_model.pth")


if __name__ == "__main__":
    main()