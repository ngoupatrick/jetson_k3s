# script edit:
# sudo vi /k3s-ext-1/code/scripts/train_resnet50.py
# Run this script with :
# python3 /mnt/code/scripts/train_resnet50.py --model_path /mnt/code/models/resnet50-0676ba61.pth --data_dir /mnt/code/images --save_path /mnt/code/glass_model_resnet50.pth --epochs 3 --batch_size 4

print("-- Import libs --")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import time
from datetime import datetime
import logging
import sys
import argparse
import os

# Configuration du logging pour Kubernetes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Script pour traiter un fichier spécifique.")
    parser.add_argument("--model_path", type=str, default="/mnt/code/models/resnet50-0676ba61.pth", help="Model weights path.")
    parser.add_argument("--data_dir", type=str, default="/mnt/code/images", help="Directory containing the images.")
    parser.add_argument("--save_path", type=str, default="/mnt/code/glass_model_resnet50.pth", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    args = parser.parse_args()
        
    if os.path.exists(args.model_path):
        logger.info(f"Model weights found at {args.model_path}")
    else:
        logger.error(f"Model weights not found at {args.model_path}")
        sys.exit(1)
        
    if os.path.exists(args.data_dir):
        logger.info(f"Data directory found at {args.data_dir}")
    else:
        logger.error(f"Data directory not found at {args.data_dir}")
        sys.exit(1)
        
    # Configuration
    print("-- Init device --")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOCAL_WEIGHTS = args.model_path
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size  # Reduit pour eviter OOM sur Jetson avec ResNet50
    DATA_DIR = args.data_dir
    SAVE_PATH = args.save_path
    
    logger.info(f"------ Batch Size {BATCH_SIZE}")
    logger.info(f"------ Epochs found at {EPOCHS}")
    logger.info(f"------ Model weights {LOCAL_WEIGHTS}")
    logger.info(f"------ Data iamges directory {DATA_DIR}")
    logger.info(f"------ Save path {SAVE_PATH}")

    # 1. Pretraitement
    logger.info(f"-- Pretraitement --")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Augmentation de donnees
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Chargement des donnees
    logger.info("-- Chargement des donnees --")
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    logger.info(f"Classes detectees : {dataset.class_to_idx}")
    num_classes = len(dataset.classes)
    logger.info(f"Nombre de classes : {num_classes}")
    # Nombre total d'images
    total_images = len(dataset)
    logger.info(f"Nombre total d'images : {total_images}")
    
    # 3. Modele ResNet50
    logger.info("-- Chargement Modele ResNet50 --")
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(LOCAL_WEIGHTS))
    # Modifier la dernière couche pour les classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 4. Boucle d'entrainement
    logger.info("-- Debut Boucle d'entrainement --")
    logger.info(f"Debut de l'entrainement sur {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        start_time = time.time()
        start_readable = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
        logger.info(f"Start time {start_readable} - epoch {epoch}...")
        i = 1

        for images, labels in train_loader:
            t_time = time.time()
            t_readable = datetime.fromtimestamp(t_time).strftime('%H:%M:%S')
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if i % 1 == 0:
                logger.info(f"loading image {i*BATCH_SIZE}, Total images: {total_images}, start_time {t_readable} - epoch {epoch}...")
            i=i+1

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        duration = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Temps: {duration:.2f}s")
        
    # 5. Sauvegarde
    logger.info(f"-- Sauvegarde {SAVE_PATH} --")
    torch.save(model.state_dict(), SAVE_PATH)
    logger.info(f"Modele sauvegarde dans {SAVE_PATH}")

if __name__ == "__main__":
            
    main()
