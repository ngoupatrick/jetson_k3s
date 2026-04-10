# script edit:
# sudo vi /k3s-ext-1/code/scripts/predict_rnn.py
# Run this script with :
# python3 /mnt/code/scripts/predict_rnn.py /mnt/code/test

print(f"-- import libs --")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
import logging

# Configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# RNN Model Structure
class ImageRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ImageRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

logger.info(f"-- Init Device --")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/mnt/code/glass_model_rnn.pth"
CLASSES = ['empty', 'plastic', 'with-beer', 'with-coffee', 'with-juice', 'with-milk', 'with-tea', 'with-water', 'with-wine']

def predict_folder(folder_path):
    # 1. Model Configuration
    logger.info(f"-- 1. Model Configuration --")
    INPUT_SIZE = 224 * 3
    SEQ_LEN = 224
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    
    model = ImageRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=len(CLASSES))
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Error : Model not found at {MODEL_PATH}")
        return

    # Load the checkpoint dictionary
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Extraction of model weights from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint (Saved at epoch {checkpoint.get('epoch')})")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE).eval()

    # 2. Transformations
    logger.info(f"-- 2. Transformations --")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Image Loop
    logger.info(f"Analyzing images in: {folder_path}")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(folder_path):
        logger.info(f"Error: Folder {folder_path} not found")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not files:
        logger.info("No images found in folder.")
        return

    with torch.no_grad():
        logger.info(f"Starting prediction loop in: {folder_path}")
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            logger.info(f"image : {filename}")
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).view(-1, SEQ_LEN, INPUT_SIZE).to(DEVICE)

                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)

                logger.info(f"  > PREDICTION : {CLASSES[pred.item()]} ({conf.item()*100:.2f}%)")

                for i, class_name in enumerate(CLASSES):
                    p = probs[0][i].item() * 100
                    logger.info(f"    - {class_name:12}: {p:.2f}%")

            except Exception as e:
                logger.info(f"Error on {filename}: {e}")

if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/code/test"
    predict_folder(target_folder)
