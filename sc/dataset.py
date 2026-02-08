import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_dataloaders(data_path, batch_size=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolder(root=data_path, transform=transform)
    # Shuffle is key to ensure classes are mixed in batches
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, dataset.classes
