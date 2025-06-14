import os
import shutil
import time
import random
import argparse
from src.utils.logger import logger

def inject(src_dir, dst_dir, delay: float, seed: int = None):
    # Gather all files into single flat list
    jobs =[]
    for cls in os.listdir(src_dir):
        cls_src = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_src):
            continue
        cls_dst = os.path.join(dst_dir, cls)
        for fname in os.listdir(cls_src):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            # Skip files that already exist in destination
            if os.path.exists(os.path.join(cls_dst, fname)):
                logger.warning(f"[Injector] Skipping {cls}/{fname} (already exists in destination)")
                continue
            src_path = os.path.join(cls_src, fname)
            dst_path = os.path.join(cls_dst, fname)
            jobs.append((src_path, dst_path))
    
    # Set seed
    if seed is not None:
        random.seed(seed)
    random.shuffle(jobs)

    # Inject files with delay
    for src_path, dst_path in jobs:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        logger.info(f"[Injector] Waiting {delay}s before injecting {cls}/{fname}")
        time.sleep(delay)
        shutil.copy(src_path, dst_path)
        logger.info(f"[Injector] Injected {cls}/{fname} -> {dst_path}")
    logger.info(f"[Injector] All files injected.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",    required=True,
                   help="Held-out test set (e.g. data/test)")
    p.add_argument("--dst_dir",    required=True,
                   help="Pipeline entry (e.g. data/raw)")
    p.add_argument("--delay",   type=float, default=5.0,
                   help="Seconds between each file injection")
    p.add_argument("--seed",   type=int, default=None,
                     help="Random seed for shuffling files")
    args = p.parse_args()
    try:
        inject(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            delay=args.delay,
            seed=args.seed
        )
    except KeyboardInterrupt:
        logger.info("[Injector] Stopped by user.")
