import os
from PIL import Image

# Paths
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def preprocess_image(input_path, output_path, size=(64, 64)):
    """
    Preprocess a single image by resizing it to the specified size and saving it.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        img = img.resize(size)
        img.save(output_path)

def preprocess_directory(raw_dir, processed_dir, size=(64, 64)):
    """
    Preprocess all images in a directory by resizing them and saving to a new directory.
    """
    os.makedirs(processed_dir, exist_ok=True)
    for class_name in os.listdir(raw_dir):
        class_raw = os.path.join(raw_dir, class_name)
        class_processed = os.path.join(processed_dir, class_name)
        if not os.path.isdir(class_raw):
            continue
        for fname in os.listdir(class_raw):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(class_raw, fname)
                output_path = os.path.join(class_processed, fname)
                preprocess_image(input_path, output_path, size)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Preprocess images by resizing them.")
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR)
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--size", type=int, nargs=2, default=(64, 64))
    
    args = parser.parse_args()
    
    preprocess_directory(args.raw_dir, args.processed_dir, size=tuple(args.size))
    