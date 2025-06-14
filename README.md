# Geospatial ML Pipeline

> **A self-contained demo** of a real-time satellite-imagery ML pipeline—from raw files to live dashboard—in under 2 weeks.

---

## Project Overview

This repo shows how to take a folder of satellite-style images, simulate streaming arrivals, train a MobileNetV2 classifier, and serve real-time predictions in a live Streamlit dashboard. You’ll get to see:

1. **Data splitting** (90 % train / 10 % hold-out)  
2. **File-watcher** that moves “raw” → “processed”  
3. **Static preprocessing** (resize & normalize)  
4. **Train script** with freeze→fine-tune schedule and loss/accuracy plots  
5. **Injector** that feeds test images at configurable intervals  
6. **Single-file inference** on arrival, logged to CSV  
7. **Streamlit dashboard** showing batch metrics and live streaming

Everything is modular, reproducible, and easy to demo—even on an HPC cluster.

---

## Repo Structure
````
├── data/
│   ├── raw/               ← watcher’s input folder
│   ├── processed/         ← static-preprocessed images
│   └── test/              ← held-out 10 % for streaming
│
├── logs/                  ← watcher, injector & training logs
├── outputs/
│   ├── models/            ← best\_model.pt, figures, class\_names.json
│   ├── predictions.csv    ← batch test results
│   └── streaming\_predictions.csv
│
├── requirements.txt       ← all non-PyTorch deps
├── train.sbatch           ← SLURM script for HPC training
├── README.md              ← you are here
│
└── src/
│     ├── ingest/
│        ├── watcher.py     ← file I/O
│        ├── infer.py       ← batch & single-file inference
│        └── ...
│     ├── preprocessing/
│        ├── split\_dataset.py
│        └── transform.py
│     ├── model/
│        ├── dataloader.py
│        └── train.py
│     ├── utils/
│        └── injector.py
│     └── visualization/
│        └── dashboard.py   ← Streamlit app
````

---

## Quickstart

### 1. Clone & Environment

```bash
git clone https://github.com/yourusername/GeoSpatial.git
cd GeoSpatial

# On local or HPC with Miniforge3:
module load Miniforge3
conda create -n geo-ml python=3.11.5 pip -y
conda activate geo-ml

# On GPU node:
module load CUDA/12.4.0
pip install torch torchvision torchaudio
pip install -r requirements.txt
````

---

### 2. Data Preparation

```bash
# 2.1 Split the EuroSAT folders (90 / 10 stratified)
python src/preprocessing/split_dataset.py

# 2.2 Watcher & static preprocessing
python -m src.ingest.watcher &       # start file‐mover
python src/preprocessing/transform.py \
     --src_dir data/raw \
     --dst_dir data/processed \
     --size 64 64
```

---

### 3. Train

```bash
# Interactive test
python -m src.model.train \
  --processed_dir data/processed \
  --output_dir outputs/models \
  --num_classes 10 \
  --batch_size 64 \
  --num_workers 4 \
  --num_epochs 30 \
  --lr 1e-3

# Or via SLURM
sbatch train.sbatch
```

After training, you’ll get:

* `outputs/models/best_model.pt`
* `outputs/models/figures/{loss,acc}_curve.png`
* `outputs/class_names.json`

---

### 4. Demo Streaming Inference & Dashboard

```bash
# 4.1 Start the watcher
python -m src.ingest.watcher &

# 4.2 Start the injector (feeds one image every N secs)
python src/utils/injector.py \
  --src_dir data/test \
  --dst_dir data/raw \
  --delay 10 \
  --seed 42 &

# 4.3 Launch the Streamlit dashboard
streamlit run src/visualization/dashboard.py
```

In the sidebar you can:

* **Set injector delay** (1–60 s)
* **Start / Stop** injector & watcher
* Pick “Batch Metrics” vs. “Streaming” pages

---

## Dashboard Features

* **Batch Metrics**: overall accuracy, confusion matrix, avg. confidence by class
* **Streaming**: live table of last 10 predictions, cumulative class counts, confidence trend, and thumbnail of the last image processed

---

## Tips & Next Steps

* **Add geocoordinates** to images and overlay on a map (Folium or Kepler.gl)
* **Containerize** with Docker + GitHub Actions for CI/CD
* **Scale** to real message queues (Kafka/PubSub) for high-throughput streams
* **Monitor** model drift by tracking per-class accuracy over time
* **Upgrade** to semantic segmentation or multispectral inputs

---

## License & Acknowledgments

This project is MIT-licensed. Inspired by the needs of satellite + energy data workflows—and built in a weekend hackathon style. Enjoy, adapt, and send feedback!
