print(f"-- import libs --")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from PIL import Image
import os
import sys
import logging
import io
from flask import Flask, request, jsonify

# Configuration du logging pour Kubernetes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info(f"-- Init Device --")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/mnt/code/glass_model_resnet50.pth"
# L'ordre doit correspondre à l'ordre alphabetique des dossiers
# {'empty': 0, 'plastic': 1, 'with-beer': 2, 'with-coffee': 3, 'with-juice': 4, 'with-milk': 5, 'with-tea': 6, 'with-water': 7, 'with-wine': 8}
CLASSES = ['empty', 'plastic', 'with-beer', 'with-coffee', 'with-juice', 'with-milk', 'with-tea', 'with-water', 'with-wine']


# 1. Configuration du modele
logger.info(f"-- 1. Configuration du modele --")
model = models.resnet50()    
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
if not os.path.exists(MODEL_PATH):
    logger.info(f"Erreur : Modele non trouve ici {MODEL_PATH}")
    sys.exit(1)
    
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE).eval()

# 2. Transformations
logger.info(f"-- 2. Transformations --")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict_folder():
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400    
    file = request.files['image']
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            
        result = {
            "prediction": CLASSES[pred.item()],
            "confidence": float(conf.item()),
            "details": {CLASSES[i]: float(probs[0][i].item()) for i in range(len(CLASSES))}
        }
        
        logger.info(f"PREDICTION : {result['prediction']} ({result['confidence']*100:.2f}%)")
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"Erreur lors de la prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Écoute sur toutes les interfaces pour K3s
    app.run(host='0.0.0.0', port=5000)