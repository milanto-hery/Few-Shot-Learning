import torch
import torch.nn.functional as F
from PIL import Image
import os
import pandas as pd


def compute_prototypes(support_features, support_labels):
    # Get unique classes present in THIS specific batch
    unique_labels = torch.unique(support_labels)
    prototypes = []
    for c in unique_labels:
        # Calculate mean vector for each class found
        class_features = support_features[support_labels == c]
        prototypes.append(class_features.mean(0))
        
    return torch.stack(prototypes), unique_labels

def prototypical_loss(prototypes, unique_labels, query_features, query_labels):
    # Calculate distance between every query and every prototype
    distances = torch.cdist(query_features, prototypes)
    
    # Map the query labels to the index of the prototype (0, 1, 2...)
    # This prevents the "Index out of bounds" crash
    label_to_idx = {label.item(): i for i, label in enumerate(unique_labels)}
    
    # Only keep queries that have a matching prototype in this batch
    valid_mask = torch.tensor([label.item() in label_to_idx for label in query_labels])
    if not valid_mask.any():
        return None

    target_indices = torch.tensor([label_to_idx[label.item()] for label in query_labels[valid_mask]]).to(query_features.device)
    
    log_p_y = F.log_softmax(-distances[valid_mask], dim=1)
    return F.nll_loss(log_p_y, target_indices)
    

def run_prototypical_inference(model, query_path, support_loader, transform, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labs in support_loader:
            if imgs.shape[0] > 0:
                features = model(imgs.to(device))
                all_features.append(features)
                all_labels.append(labs)
    
    if len(all_features) == 0:
        raise ValueError("Support loader is empty! Check your 'data/train' path and folder structure.")
            
    support_features = torch.cat(all_features)
    support_labels = torch.cat(all_labels).to(device)
    
    # Compute prototypes
    from src.engine import compute_prototypes
    prototypes, unique_labs = compute_prototypes(support_features, support_labels)
    
    # Process Query
    q_img_raw = Image.open(query_path).convert("RGB")
    q_tensor = transform(q_img_raw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_feature = model(q_tensor)
        distances = torch.cdist(query_feature, prototypes)
        best_idx = torch.argmin(distances).item()
    
    return {
        "class": support_loader.dataset.classes[unique_labs[best_idx].item()].upper(),
        "distance": distances[0][best_idx].item(),
        "raw_img": q_img_raw
    }
    


def batch_test_folder(model, test_dir, support_loader, transform, device):
    """
    Runs prediction on every image in the test folder and returns a summary.
    """
    model.eval()
    results = []
    
    # Get all images in test folder
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Starting batch test on {len(test_files)} images...")
    
    for filename in test_files:
        query_path = os.path.join(test_dir, filename)
        # Reuse our inference logic
        prediction = run_prototypical_inference(model, query_path, support_loader, transform, device)
        
        results.append({
            "filename": filename,
            "prediction": prediction["class"],
            "distance": round(prediction["distance"], 4)
        })
    
    return pd.DataFrame(results)
    
    
    
