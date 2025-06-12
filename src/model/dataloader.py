import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT processed images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.images= []     # List of [path, class_index]
        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append((os.path.join(class_dir, fname), idx))
        self.transform = transform
    
    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, idx):
        path, label = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(processed_dir, batch_size=32, val_split=0.1, num_workers=4):
    """
    Create training and validation DataLoaders with transforms and split.
    """
    # Define transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Full dataset for splitting
    full_dataset = EuroSATDataset(processed_dir, transform=train_transform)
    labels = [label for _, label in full_dataset.images]
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=42
    )

    train_base = EuroSATDataset(processed_dir, transform=train_transform)
    val_base = EuroSATDataset(processed_dir, transform=val_transform)

    # Create subsets
    train_subset = Subset(train_base, train_idx)
    # train_subset.dataset.transform = train_transform
    val_subset = Subset(val_base, val_idx)
    # val_subset.dataset.transform = val_transform

    # Check if subsets have correct transforms
    assert train_subset.dataset.transform == train_transform, "Train subset transform mismatch"
    assert val_subset.dataset.transform == val_transform, "Validation subset transform mismatch"

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


