import cv2
from ultralytics import YOLO
import math

# --- 1. Muat Model ---
model = YOLO('best.pt')

# --- 2. Nama Kelas ---
class_names = ["paper", "rock", "scissors"]

# --- 3. Buka Kamera ---
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# --- 4. Loop Deteksi Real-Time ---
while True:
    # Baca frame dari kamera
    success, frame = cap.read()
    
    if not success:
        print("Error: Tidak bisa membaca frame.")
        break

    # --- 5. Jalankan Deteksi YOLO ---
    results = model(frame, stream=True)

    # --- 6. Proses Hasil Deteksi ---
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Dapatkan koordinat kotak
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Dapatkan confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100
            
            # Dapatkan nama kelas
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]

            # --- 7. Gambar Kotak dan Label di Frame ---
            
            # Buat label (Nama Kelas + Confidence)
            label = f'{class_name}: {confidence}'
            
            # Warna untuk kotak (B, G, R)
            color = (255, 0, 255) # Magenta

            # Gambar kotak (bounding box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Gambar latar belakang untuk teks
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 15, y1), color, -1)
            
            # Tambahkan teks label
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # --- 8. Tampilan Hasil ---
    cv2.imshow('YOLOv8 Webcam - Deteksi Batu Gunting Kertas', frame)

    # --- 9. Tombol Keluar ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 10. clear ---
cap.release()
cv2.destroyAllWindows()
print("Kamera ditutup.")