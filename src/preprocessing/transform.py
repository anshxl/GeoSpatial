import os
from PIL import Image

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

