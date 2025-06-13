import os
import time
import csv
import json
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from torchvision import transforms as T
from src.utils.logger import logger
from src.ingest.infer import get_class_names, build_model, infer_single_image

# Define constants
CLASS_NAMES = get_class_names(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/class_names.json")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = build_model(len(CLASS_NAMES),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../outputs/models/best_model.pt")),
                    DEVICE)
TRANS = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CSV_PATH = "outputs/streaming_predictions.csv"
# ensure CSV exists with header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath","true_label","pred_label","confidence"])

# Configure path relative to project root
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),  "../../data/processed"))

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle new file creation events."""
        if event.is_directory:
            return
        src_path = event.src_path
        if src_path.lower().endswith((".jpeg", '.jpg', '.png')):
            filename = os.path.basename(src_path)
            dest_path = os.path.join(PROCESSED_DIR, filename)
            try:
                shutil.move(src_path, dest_path)
                logger.info(f"Moved {filename} to processed directory.")

                # Perform inference
                pred, conf = infer_single_image(dest_path, MODEL, TRANS, CLASS_NAMES, DEVICE)
                true_label = dest_path.split(os.path.sep)[-1].split('_')[0] # Assuming the filename is the true label
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([dest_path, true_label, pred, f"{conf:.4f}"])
                
                logger.info(
                    f"Inferred {filename} → {pred} (conf={conf:.2f})"
                )
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
        
def main():
    # Ensure directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Set up event handler and observer
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=RAW_DIR, recursive=True)
    observer.start()
    logger.info(f"Watcher started. Monitoring {RAW_DIR} for new files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()