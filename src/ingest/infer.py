import os
import csv
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

from src.utils.logger import logger

def get_class_names(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    
def build_model(num_classes, model_path, device):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, num_classes)

    # Load trained weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model

def infer(args):
    logger.info("Starting inference...")
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    # Load class names
    class_names = get_class_names(args.class_names_json)
    num_classes = len(class_names)
    logger.info(f"Loaded {num_classes} classes: {class_names}")

    # Build model
    model = build_model(num_classes, args.model_path, device)

    # Define image transformations
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare output CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'true_label', 'pred_label', 'confidence'])

        # walk through subdirectories
        for true_label in class_names:
            cls_dir = os.path.join(args.processed_dir, true_label)
            if not os.path.isdir(cls_dir):
                logger.warning(f"No folder for class {true_label} in {args.processed_dir}")
                continue

            for filename in os.listdir(cls_dir):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                fpath = os.path.join(cls_dir, filename)
                try:
                    img = Image.open(fpath).convert('RGB')
                    inp = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(inp)
                        probs = nn.functional.softmax(output, dim=1)
                        conf, pred = probs.max(dim=1)

                    writer.writerow([
                        fpath,
                        true_label,
                        class_names[pred.item()],
                        f"{conf.item()}:.4f"
                    ])
                except Exception as e:
                    logger.error(f"Failed inference on {fpath}", exc_info=True)

    logger.info(f"Inference completed. Results saved to {args.output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir",     type=str, required=True,
                   help="Folder of class-subdirs with images to infer on")
    p.add_argument("--class_names_json",  type=str, required=True,
                   help="Path to outputs/class_names.json created during training")
    p.add_argument("--model_path",        type=str, required=True,
                   help="Path to your .pt checkpoint")
    p.add_argument("--output_csv",        type=str, default="outputs/predictions.csv",
                   help="Where to write filepath,true_label,pred_label,confidence")
    p.add_argument("--device",            type=str, default=None,
                   help="‘cuda’ or ‘cpu’; auto-selects if omitted")
    args = p.parse_args()
    infer(args)
