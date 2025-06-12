import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from src.model.dataloader import get_dataloaders
from src.utils.logger import logger

def set_backbone_requires_grad(model, requires_grad: bool):
    """
    Set requires_grad for all parameters in the model's backbone.
    """
    for param in model.features.parameters():
        param.requires_grad = requires_grad

def train_model(
        processed_dir,
        output_dir,
        num_classes,
        batch_size=32,
        val_split=0.1,
        num_workers=4,
        num_epochs=20,
        learning_rate=0.001,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    FREEZE_EPOCHS = num_epochs // 2
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training on device: {device}")

    # Get loaders
    train_loader, val_loader = get_dataloaders(
        processed_dir,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers
    )
    
    # Initialize model
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # Freeze backbone parameters
    set_backbone_requires_grad(model, requires_grad=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    best_accuracy = 0.0

    for epoch in range(1, num_epochs+1):
        # Unfreeze backbone parameters after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            set_backbone_requires_grad(model, requires_grad=True)
            logger.info(f"Unfreezing backbone parameters at epoch {epoch}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        logger.info(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_corrects / val_total
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save the model if it has the best accuracy so far
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved: {save_path}")
        
        scheduler.step()
    logger.info(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--processed_dir", default='data/processed')
    parser.add_argument("--output_dir", default='output/models')
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train_model(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.lr
    )
