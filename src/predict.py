import torch
import argparse
import os
import sys
from PIL import Image

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.model import PrimatePrototypicalNet
from src.dataset import get_dataloaders
from src.engine import run_prototypical_inference, batch_test_folder

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Load Model
    model = PrimatePrototypicalNet().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights: {args.model_path}")
    else:
        print(f"Warning: Checkpoint {args.model_path} not found. Running with baseline weights.")

    # Setup Support Context
    support_loader, _ = get_dataloaders(args.support_data, batch_size=args.support_size)
    transform = support_loader.dataset.transform

    # Inference Execution
    if os.path.isdir(args.query):
        results = batch_test_folder(model, args.query, support_loader, transform, device)
        print("\n--- Batch Inference Summary ---")
        print(results.to_string(index=False))
    else:
        prediction = run_prototypical_inference(model, args.query, support_loader, transform, device)
        print(f"\n--- Prediction Results ---")
        print(f"Target Image: {os.path.basename(args.query)}")
        print(f"Classified As: {prediction['class']}")
        print(f"Euclidean Distance: {prediction['distance']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Metric-Based Inference Script")
    parser.add_argument("--query", type=str, required=True, help="Image or directory to analyze")
    parser.add_argument("--support-data", type=str, default="data_lemur/train", help="Support set for prototypes")
    parser.add_argument("--model-path", type=str, default="primate_model.pth", help="Checkpoint file path")
    parser.add_argument("--support-size", type=int, default=20, help="Support samples context")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    main(args)
