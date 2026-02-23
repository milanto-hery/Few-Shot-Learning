import torch
import torch.optim as optim
import argparse
import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.path.abspath("."))

from src.model import PrimatePrototypicalNet
from src.dataset import get_dataloaders
from src.engine import compute_prototypes, prototypical_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Executing training on: {device}")

    # 1. Setup Data
    train_loader, _ = get_dataloaders(args.data, batch_size=args.batch_size)

    # 2. Setup Model & Optimizer
    model = PrimatePrototypicalNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    model.train()
    for epoch in range(args.episodes):
        total_loss = 0
        valid_batches = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            embeddings = model(images.to(device))

            # Split Support/Query
            half = len(labels) // 2
            if half == 0: continue
            
            support_feats, query_feats = embeddings[:half], embeddings[half:]
            support_labs, query_labs = labels[:half], labels[half:]

            prototypes, unique_labs = compute_prototypes(support_feats, support_labs.to(device))
            loss = prototypical_loss(prototypes, unique_labs, query_feats, query_labs.to(device))

            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                valid_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss/max(1, valid_batches)
            print(f"Episode {epoch+1}/{args.episodes} | Loss: {avg_loss:.6f}")

    # 4. Save Artifact
    torch.save(model.state_dict(), args.output)
    print(f"Model checkpoint persisted to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episodic Training Engine for Few-Shot Learning")
    parser.add_argument("--data", type=str, default="data_lemur/train", help="Dataset directory path")
    parser.add_argument("--episodes", type=int, default=35, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Images per batch")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--output", type=str, default="primate_model.pth", help="Path for saved weights")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU execution")
    
    args = parser.parse_args()
    main(args)
