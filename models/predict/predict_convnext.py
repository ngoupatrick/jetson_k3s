print(f"-- import libs --")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/mnt/code/glass_model_convnext.pth"
CLASSES = ['empty', 'plastic', 'with-beer', 'with-coffee', 'with-juice', 'with-milk', 'with-tea', 'with-water', 'with-wine']

def predict_folder(folder_path):
    # 1. Configuration du modele
    logger.info(f"-- 1. Configuration du modele ConvNeXt Tiny --")
    model = models.convnext_tiny(weights=None)    
    
    # Doit correspondre a la structure modifiee lors de l'entrainement
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASSES))

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Erreur : Modele non trouve ici {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    # 2. Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Boucle sur les images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    with torch.no_grad():
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)

                logger.info(f"  > PREDICTION : {CLASSES[pred.item()]} ({conf.item()*100:.2f}%)")
            except Exception as e:
                logger.error(f"Erreur sur {filename}: {e}")

if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/code/test"
    predict_folder(target_folder)
