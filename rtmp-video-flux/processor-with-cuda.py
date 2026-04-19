import gi
import numpy as np
import os
import sys
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import signal
import logging
import argparse
import time

is_up = False

# Desactive les messages WARN et FIXME de GStreamer pour un log propre
os.environ["GST_DEBUG"] = "0"

# Configuration du logging pour Kubernetes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

# --- Gestion des arguments ---
parser = argparse.ArgumentParser(description="Script lisant un flux RTSP et effectuant une inference avec TensorRT.")
parser.add_argument("--stream_key", type=str, default="host_stream", help="Stream key for the RTSP feed")
parser.add_argument("--engine_path", type=str, default="/mnt/yolo/models/yolov8n.engine", help="Path to the TensorRT engine file")
args = parser.parse_args()
if os.path.exists(args.engine_path):
    logger.info(f"Engine found at {args.engine_path}")
else:
    logger.error(f"Engine not found at {args.engine_path}")
    sys.exit(1)

STREAM_KEY = args.stream_key
ENGINE_PATH = args.engine_path

# --- Gestion du signal Ctrl+C ---
running = True
def signal_handler(sig, frame):
    global running
    logger.info("Signal d'arret recu (Ctrl+C)...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# --- Configuration TensorRT ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_engine(path):
    if not os.path.exists(path):
        logger.error("Engine introuvable a {}".format(path))
        sys.exit(1)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

logger.info("Initialisation TensorRT avec le modele: {}".format(ENGINE_PATH))
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

bindings = []
input_mem = None
output_mem = None

for binding in engine:
    shape = engine.get_binding_shape(binding)
    size = trt.volume(shape)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        input_mem = {"host": host_mem, "device": device_mem, "shape": shape}
    else:
        output_mem = {"host": host_mem, "device": device_mem, "shape": shape}

# --- Configuration GStreamer / RTSP ---
MM_IP = os.getenv("MEDIAMTX_SERVICE_PORT_8554_TCP_ADDR", "10.43.7.10")
MM_PORT = os.getenv("MEDIAMTX_SERVICE_PORT_8554_TCP_PORT", "8554")
RTSP_URL = "rtsp://{}:{}/{}".format(MM_IP, MM_PORT, STREAM_KEY)

pipeline_str = (
    "rtspsrc location={} latency=100 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink name=sink drop=True max-buffers=1 emit-signals=True"
).format(RTSP_URL)

pipeline = Gst.parse_launch(pipeline_str)
sink = pipeline.get_by_name("sink")

try:
    while running:
        if not is_up:
            # Reconstruction du pipeline pour nettoyer les sockets rtspsrc
            if 'pipeline' in locals():
                pipeline.set_state(Gst.State.NULL)
            pipeline = Gst.parse_launch(pipeline_str)
            sink = pipeline.get_by_name("sink")
            
            pipeline.set_state(Gst.State.PLAYING)
            ret, state, pending = pipeline.get_state(5 * Gst.SECOND)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                is_up = False
                logger.error("Impossible de se connecter au flux RTSP {}. Nouvelle tentative dans 5s...".format(RTSP_URL))
                pipeline.set_state(Gst.State.NULL)
                is_up = False
                time.sleep(5)
                continue
            elif state != Gst.State.PLAYING:
                continue
            else:
                if is_up == False:
                    logger.info("Connexion au flux RTSP reussie: {}".format(RTSP_URL))
                    is_up = True

        sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
        if not sample:
            # Si on ne reçoit rien, on vérifie si le pipeline est tombé
            logger.error("Flux perdu ou injoignable. Reconnexion...")
            pipeline.set_state(Gst.State.NULL)
            is_up = False
            time.sleep(5)
            continue

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w, h = struct.get_value('width'), struct.get_value('height')

        result, map_info = buf.map(Gst.MapFlags.READ)
        if result:
            frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8)
            
            blob = cv2.resize(frame, (640, 640)).transpose(2, 0, 1).astype(np.float32)
            blob /= 255.0
            
            np.copyto(input_mem["host"], blob.ravel())
            cuda.memcpy_htod(input_mem["device"], input_mem["host"])
            context.execute_v2(bindings=bindings)
            cuda.memcpy_dtoh(output_mem["host"], output_mem["device"])
            
            output = output_mem["host"].reshape(output_mem["shape"])
            output = np.squeeze(output).transpose()
            
            x_factor, y_factor = w / 640, h / 640

            for detection in output:
                scores = detection[4:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    label = CLASSES[class_id]
                    xc, yc, nw, nh = detection[0], detection[1], detection[2], detection[3]
                    
                    x_min = int((xc - nw / 2) * x_factor)
                    y_min = int((yc - nh / 2) * y_factor)
                    box_w, box_h = int(nw * x_factor), int(nh * y_factor)

                    logger.info("Detecte: {} | Conf: {:.2f} | Loc: x={}, y={}, w={}, h={}".format(
                        label, confidence, x_min, y_min, box_w, box_h))
            
            buf.unmap(map_info)

except Exception as e:
    logger.error("Erreur durant l'execution: {}".format(e))
finally:
    logger.info("Fermeture du pipeline et liberation des ressources...")
    pipeline.set_state(Gst.State.NULL)
    sys.exit(0)
