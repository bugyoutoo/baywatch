import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

TEST_DIR = "data/test"
MODEL_PATH = "model.pth"
THRESHOLD = 0.5

# Transforms must match training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

alerts_triggered = 0
results = []

# Loop through all .png files in TEST_DIR
for file in os.listdir(TEST_DIR):
    if file.endswith(".png"):
        img_path = os.path.join(TEST_DIR, file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output).squeeze().numpy()

        debris_score = float(probs[0])
        cloud_score = float(probs[1])

        print(f"{file}: Debris={debris_score:.2f}, Cloud={cloud_score:.2f}")

        debris_detected = debris_score >= THRESHOLD
        if debris_detected:
            alerts_triggered += 1
            print("🚨 ALERT: Cleanup needed!")

        results.append({
            "image": file,
            "debris_detected": debris_detected,
            "confidence": round(debris_score * 100, 2),
            "status": "Needs Review" if debris_detected else "Safe"
        })

print(f"\nInference complete. Alerts triggered: {alerts_triggered}")

# Ensure dashboard folder exists
os.makedirs("dashboard", exist_ok=True)

# Save results to JSON file
with open("dashboard/inference_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Inference results saved to dashboard/inference_results.json")
