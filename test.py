import time
from ultralytics import YOLO
import numpy as np

drive_project_path = '/content/drive/MyDrive/yolo'

# --- 1. Data.yaml ---

data_yaml_path = "rock-paper-scissors-1/data.yaml"

# --- 2. Memuat Model ---
model_path = f'{drive_project_path}/rps_training_run_1/weights/best.pt'

try:
    model = YOLO(model_path)
    print(f"Berhasil memuat model dari GDrive: {model_path}")
except Exception as e:
    print(f"Error: Tidak bisa memuat model. {e}")
    exit()

# --- 3. Mulai Evaluasi (Testing) ---
print("Memulai evaluasi pada TEST SET...")
start_time = time.time()

# Panggil model.val() dengan split='test'
metrics = model.val(
    data=data_yaml_path,
    split='test',
    plots=True,
    project=drive_project_path,  
    name='rps_testing_run_1'
)

end_time = time.time()
testing_time_seconds = end_time - start_time

print(f"--- Evaluasi Selesai ---")
print(f"Total Waktu Testing (Evaluasi): {testing_time_seconds:.2f} detik")

# --- 4. Menampilkan Hasil Metrik
print("\n--- 📊 HASIL METRIK EVALUASI (TEST SET) ---")

# 1. mAP
print("\n### 1. mAP (Mean Average Precision)")
print(f"   mAP50-95 (utama): {metrics.box.map:.4f}")
print(f"   mAP50 (populer): {metrics.box.map50:.4f}")

# 2. Precision, Recall, F1-Score
print("\n### 2. Precision, Recall, F1-Score")
p = metrics.box.mp
r = metrics.box.mr
print(f"   Precision (Rata-rata): {p:.4f}")
print(f"   Recall (Rata-rata): {r:.4f}")

# Hitung F1-Score secara manual
if (p + r) > 0:
    f1 = 2 * (p * r) / (p + r)
    print(f"   F1-Score (Dihitung manual): {f1:.4f}")
else:
    print("   F1-Score: 0.0 (Precision atau Recall adalah 0)")

# 3. ROC/AUC
print("\n### 3. ROC/AUC (Kurva P-R)")
print("   - Untuk Object Detection, metrik yang setara adalah P-R Curve (Precision-Recall Curve).")
print(f"   - Nilai 'mAP' (misal {metrics.box.map:.4f}) adalah 'Area di Bawah Kurva' P-R tersebut.")
print(f"   - Plot P-R curve disimpan sebagai 'PR_curve.png' di folder: {metrics.save_dir}")

# 4. Computation Time
print("\n### 4. Computation Time")
print(f"   Waktu Training: (Lihat hasil skrip training sebelumnya)")
print(f"   Waktu Testing: {testing_time_seconds:.2f} detik")

print("\n---")
print(f"Semua hasil dan plot evaluasi disimpan di: {metrics.save_dir}")