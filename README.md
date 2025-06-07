# Geospatial ML Pipeline

This project simulates a real-time machine learning pipeline that processes satellite imagery to generate land-use predictions and visualize them on an interactive dashboard.

## 🌍 Use Case
Imagine a system that ingests satellite or aerial imagery in real-time and uses ML to classify terrain types—farmland, forests, urban areas, water bodies—and displays these insights for energy or maritime applications.

## 📁 Project Structure
```
project_root/
├── data/                  # Raw and processed satellite images
├── src/                  # Ingestion, preprocessing, ML model code
├── app/                  # Streamlit dashboard
├── outputs/              # Saved predictions and trained models
├── notebooks/            # EDA and experiments
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and usage
└── run_pipeline.py       # Entry point for pipeline simulation
```

## 🚀 Quick Start
1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/geospatial-ml-pipeline.git
   cd geospatial-ml-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**
   - [EuroSAT RGB Dataset (27,000+ satellite images)](https://github.com/phelber/EuroSAT)
   - Place them under `data/raw/`

4. **Run the Pipeline**
   ```bash
   python run_pipeline.py
   ```

5. **Launch the Dashboard**
   ```bash
   streamlit run app/dashboard.py
   ```

## 🔧 Features
- Watches for new imagery (simulated ingestion)
- CNN-based classification (e.g., MobileNetV2)
- Geo-tagged output with timestamped predictions
- Real-time dashboard for visualization

## 📦 Dependencies
- Python, TensorFlow/PyTorch, OpenCV, GeoPandas, Streamlit, Watchdog

## 🧠 Author & License
Developed by Anshul Srivastava 
MIT License

---

> This project is a portfolio demo meant to simulate real-world applications in energy and satellite data science.
