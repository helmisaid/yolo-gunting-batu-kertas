# uncomment jika belum diinstall
# !pip install ultralytics roboflow

import time
import os
from roboflow import Roboflow
from ultralytics import YOLO
from google.colab import drive

drive.mount('/content/drive')

# --- 1. Download Dataset dari Roboflow ---
# Ganti dengan API key dan detail project anda pada roboflow
if not os.path.exists("rock-paper-scissors-1"):
    print("Mendownload dataset dari Roboflow...")
    rf = Roboflow(api_key="API_KEY") # ganti api key roboflow 
    project = rf.workspace("simple-test").project("rock-paper-scissors-sxsw-gfczs")
    version = project.version(1)
    dataset = version.download("yolov8")
else:
    print("Dataset sudah ada.")

drive_project_path = '/content/drive/MyDrive/yolo'

data_yaml_path = os.path.join(os.getcwd(), "rock-paper-scissors-1/data.yaml")


# --- 2. Persiapan Model ---
model = YOLO('yolov8n.pt')

# --- 3. Mulai Training ---
print("Memulai training...")
start_time = time.time()

# Panggil model.train() dengan parameter
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,

    # 2. Augmentasi
    fliplr=0.5,         # Horizontal Flip
    degrees=0.2,        # Rotation
    shear=0.2,          # Shear
    scale=0.2,          # Zoom (scale 0.8x - 1.2x)
    flipud=0.0,
    mosaic=0.0,

    plots=True,

    project=drive_project_path,    
    name='rps_training_run_1'
)

end_time = time.time()
training_time_seconds = end_time - start_time

print(f"--- Training Selesai ---")
print(f"Total Waktu Training: {training_time_seconds:.2f} detik")
print(f"Hasil training disimpan di: {results.save_dir}")