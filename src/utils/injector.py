import os
import shutil
import time
import argparse
from src.utils.logger import logger

def inject(src_dir, dst_dir, delay: float):
    for cls in sorted(os.listdir(src_dir)):
        cls_src = os.path.join(src_dir, cls)
        cls_dst = os.path.join(dst_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)

        for fname in sorted(os.listdir(cls_src)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src_path = os.path.join(cls_src, fname)
            dst_path = os.path.join(cls_dst, fname)
            
            logger.info(f"[Injector] Waiting {delay}s before injecting {cls}/{fname}")
            time.sleep(delay)
            shutil.copy(src_path, dst_path)
            logger.info(f"[Injector] Injected {cls}/{fname} -> {dst_path}")
    logger.info(f"[Injector] All files injected.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",    required=True,
                   help="Held-out test set (e.g. data/split/test)")
    p.add_argument("--dst_dir",    required=True,
                   help="Pipeline entry (e.g. data/raw)")
    p.add_argument("--delay",   type=float, default=5.0,
                   help="Seconds between each file injection")
    args = p.parse_args()
    inject(args.src_dir, args.dst_dir, args.delay)