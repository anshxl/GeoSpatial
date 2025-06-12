import os
import csv
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

from src.utils.logger import logger

def get_class_names(processed_dir):
    return sorted(
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    )

def build_model(num_classes, model_path, device):
    # Load the pre-trained MobileNetV2 model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
