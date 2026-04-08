print("-- import libs --")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import sys
import logging

# Configuration du logging pour Kubernetes (ASCII)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("-- Init Device --")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemin vers votre modele entraine avec MobileNet
MODEL_PATH = "/mnt/code/glass_model_mobilenet.pth"

# L'ordre doit correspondre a l'ordre alphabetique des dossiers
CLASSES = ['empty', 'plastic', 'with-beer', 'with-coffee', 'with-juice', 'with-milk', 'with-tea', 'with-water', 'with-wine']

def predict_folder(folder_path):
    # 1. Configuration du modele MobileNetV3
    logger.info("-- 1. Configuration du modele MobileNetV3 --")
    
    # Initialisation de l'architecture
    model = models.mobilenet_v3_large(pretrained=False)
    
    # Modification du classifier pour correspondre au nombre de classes
    # Sur MobileNetV3 Large, le classifier est un bloc Sequential
    # La couche lineaire finale est model.classifier[3]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Erreur : Modele non trouve ici {MODEL_PATH}")
        return

    # Chargement des poids
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    logger.info(f"Modele charge avec succes depuis {MODEL_PATH}")

    # 2. Transformations
    logger.info("-- 2. Transformations --")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Boucle sur les images
    logger.info(f"Analyse des images dans : {folder_path}")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(folder_path):
        logger.error(f"Dossier non trouve : {folder_path}")
        return

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not files:
        logger.info("Aucune image trouvee dans le dossier.")
        return

    with torch.no_grad():
        logger.info(f"Debut des predictions...")
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)

                logger.info(f"  > IMAGE : {filename}")
                logger.info(f"    RESULTAT : {CLASSES[pred.item()]} ({conf.item()*100:.2f}%)")

                # Affichage des probabilites pour chaque classe
                for i, class_name in enumerate(CLASSES):
                    p = probs[0][i].item() * 100
                    if p > 1.0: # Affiche uniquement les probabilites > 1% pour la clarte
                        logger.info(f"    - {class_name:12}: {p:.2f}%")

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {filename}: {e}")

if __name__ == "__main__":
    # Dossier par defaut /mnt/code/test si aucun argument n'est fourni
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/code/test"
    predict_folder(target_folder)
