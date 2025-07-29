import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils import MarineDebrisDataset
import os
import csv
import numpy as np
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# === Configuration ===
MODEL_PATH = f"model_{timestamp}.pth"
METRICS_FILE = f"training_metrics_{timestamp}.csv"
TRAIN_IMG_DIR = "data/train"
TEST_IMG_DIR = "data/test"
TRAIN_CSV_FILE = "data/labels_parsed.csv"
# MODEL_PATH = "model.pth"
# METRICS_FILE = "training_metrics.csv"
BATCH_SIZE = 8
EPOCHS = 12
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# === Dataset & Dataloader ===
train_dataset = MarineDebrisDataset(TRAIN_IMG_DIR, TRAIN_CSV_FILE, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model (pretrained ResNet18) ===
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 outputs: debris, cloud
model = model.to(DEVICE)

# === Loss & Optimizer ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training ===
print("🔧 Training started...")
model.train()

# Open CSV file to write metrics
with open(METRICS_FILE, 'w', newline='') as csvfile:
    metricswriter = csv.writer(csvfile)
    metricswriter.writerow(['Epoch', 'Loss'])

    for epoch in range(EPOCHS):
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Write metrics to CSV
        metricswriter.writerow([epoch+1, total_loss])

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Training metrics saved to {METRICS_FILE}")

# === Testing ===
print("\nTesting started...")
model.eval()  # Set model to evaluation mode

# Load test dataset (without labels)
test_dataset = MarineDebrisDataset(TEST_IMG_DIR, None, transform=transform)  # Pass None for labels
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test the model
with torch.no_grad():  # No need to track gradients during testing
    for images, _ in test_loader:  # Ignore labels
        images = images.to(DEVICE)
        outputs = model(images)
        
        # Calculate probabilities
        probs = torch.sigmoid(outputs)
        
        # Get debris probabilities and predictions
        debris_probs = probs[:, 0]  # First column is debris
        debris_preds = (debris_probs > THRESHOLD).float()
        
        # Move tensors to CPU and convert to numpy
        debris_probs_np = debris_probs.cpu().numpy()
        debris_preds_np = debris_preds.cpu().numpy()
        
        # Print results for each image
        for i in range(len(debris_preds_np)):
            img_file = test_dataset.filenames[i]  # Get the filename
            debris_detected = debris_preds_np[i] == 1
            confidence = debris_probs_np[i] * 100  # Convert to percentage
            
            print(f"Image: {img_file}")
            print(f"Debris Detected: {debris_detected}")
            print(f"Confidence: {confidence:.2f}%")
            print()

print("\n✅ Testing completed!")