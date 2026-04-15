import cv2
import os
import sys
from ultralytics import YOLO

# Configuration via variables d'environnement
RTSP_URL = os.getenv("RTSP_URL", "rtsp://mediamtx-service:8554/host_stream")
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/yolo/models/yolov8n.pt")

# Pipeline GStreamer optimisé pour Jetson Nano (Hardware Decoding)
# On utilise omxh264dec pour le H264 et nvvidconv pour le transfert vers CUDA
gst_pipeline = (
    f"rtspsrc location={RTSP_URL} latency=100 ! "
    "rtph264depay ! h264parse ! omxh264dec ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
)

#gst_pipeline = (
#    f"rtmpsrc location={RTMP_URL} ! "
#    "flvdemux ! h264parse ! nvv4l2decoder ! "
#    "nvvidconv ! video/x-raw, format=BGRx ! "
#    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
#)


def main():
    print(f"--- Initialisation YOLOv8 sur GPU (FP16) ---")
    
    # 1. Chargement du modèle sur le GPU
    try:
        model = YOLO(MODEL_PATH).to('cuda')
        # On force une première passe pour charger les kernels CUDA
        model.predict(source=None, imgsz=320, half=True)
        print(f"Modèle {MODEL_PATH} chargé avec succès en FP16.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        sys.exit(1)

    # 2. Capture Vidéo
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le flux RTSP. Vérifiez MediaMTX et l'URL.")
        sys.exit(1)

    print(f"Traitement du flux : {RTSP_URL}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin du flux ou erreur de lecture.")
                break

            # 3. Inférence YOLOv8 optimisée
            # imgsz=320 réduit la charge, half=True active le FP16
            results = model(frame, stream=True, half=True, imgsz=320, verbose=False, device=0)

            for r in results:
                if len(r.boxes) > 0:
                    # Log minimaliste pour K3s
                    print(f"Détection : {len(r.boxes)} objets trouvés", end="\r")

    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    finally:
        cap.release()
        print("Ressources libérées.")

if __name__ == "__main__":
    main()
