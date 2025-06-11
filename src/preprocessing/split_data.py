import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
ORIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/EuroSAT'))
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/test'))

def split_and_copy(orig_dir=ORIG_PATH, raw_path=RAW_DIR, test_path=TEST_DIR, test_size=0.1, random_seed=42):
    filepaths = []
    labels = []
    for cls in os.listdir(orig_dir):
        cls_path = os.path.join(orig_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for filename in os.listdir(cls_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_path, filename))
                labels.append(cls)
    
    # Stratified split
    train_files, test_files, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_size, stratify=labels, random_state=random_seed
    )

    # Copy files into data/raw
    for src, cls in zip(train_files, train_labels):
        dst_dir = os.path.join(raw_path, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)
    
    # Copy files into data/test
    for src, cls in zip(test_files, test_labels):
        dst_dir = os.path.join(test_path, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)
    
    print(f"Data split complete. {len(train_files)} training files and {len(test_files)} test files copied.")

if __name__ == "__main__":
    split_and_copy()