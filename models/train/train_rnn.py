# script edit:
# sudo vi /k3s-ext-1/code/scripts/train_rnn.py
# Run this script with :
# python3 /mnt/code/scripts/train_rnn.py --data_dir /mnt/code/images --save_path /mnt/code/glass_model_rnn.pth --epochs 3 --batch_size 4

print("-- Import libs --")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import time
from datetime import datetime
import logging
import sys
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser(description="Script to train RNN.")
    parser.add_argument("--data_dir", type=str, default="/mnt/code/images", help="Directory containing the images.")
    parser.add_argument("--save_path", type=str, default="/mnt/code/glass_model_rnn.pth", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    args = parser.parse_args()
        
    if os.path.exists(args.data_dir):
        logger.info(f"Data directory found at {args.data_dir}")
    else:
        logger.error(f"Data directory not found at {args.data_dir}")
        sys.exit(1)
        
    # Configuration
    print("-- Init device --")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATA_DIR = args.data_dir
    SAVE_PATH = args.save_path
    
    # Updated RNN Specifics
    INPUT_SIZE = 224 * 3
    SEQ_LEN = 224
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4

    logger.info(f"------ Batch Size {BATCH_SIZE}")
    logger.info(f"------ Epochs found at {EPOCHS}")
    logger.info(f"------ Data images directory {DATA_DIR}")
    logger.info(f"------ Save path {SAVE_PATH}")

    # 1. Preprocessing
    logger.info(f"-- Preprocessing --")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 2. Data Loading
    logger.info("-- Loading data --")
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.classes)
    logger.info(f"Number of classes : {num_classes}")
    
    # 3. RNN Model Init
    logger.info("-- Initializing RNN Model --")
    model = ImageRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    logger.info("-- Start Training Loop --")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        start_time = time.time()
        start_readable = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
        logger.info(f"Start time {start_readable} - epoch {epoch}...")
        i = 1

        for images, labels in train_loader:
            s_readable = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
            images = images.view(-1, SEQ_LEN, INPUT_SIZE).to(DEVICE)
            labels = labels.to(DEVICE)           

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 5 == 0:
                logger.info(f"Batch {i}, Start time {s_readable} - Loss: {loss.item():.4f} - Epoch {epoch}")
            i += 1

        epoch_loss = running_loss / len(train_loader)
        duration = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Time: {duration:.2f}s")
        
    # 5. Save Model and Optimizer
    logger.info(f"-- Saving to {SAVE_PATH} --")
    checkpoint = {
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    torch.save(checkpoint, SAVE_PATH)
    logger.info(f"Checkpoint saved in {SAVE_PATH}")

if __name__ == "__main__":
    main()
