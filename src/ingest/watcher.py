import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.utils.logger import logger

# Configure path relative to project root
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),  "../../data/processed"))

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        src_path = event.src_path
        if src_path.lower().endswith((".jpeg", '.jpg', '.png')):
            filename = os.path.basename(src_path)
            dest_path = os.path.join(PROCESSED_DIR, filename)
            try:
                shutil.move(src_path, dest_path)
                logger.info(f"Moved {filename} to processed directory.")
            except Exception as e:
                logger.error(f"Error moving {filename}: {e}")
            
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