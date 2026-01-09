from PIL import Image
import torch
import cv2
from torchvision import transforms
from model import AIDetector
from fft import fft_features

# Device
device = torch.device("cpu")

# Load model
model = AIDetector().to(device)
model.load_state_dict(torch.load("detector.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load image
img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
img = transform(img).unsqueeze(0).to(device)


# FFT features
freq = fft_features(img).to(device)

# Prediction
with torch.no_grad():
    logits = model(img, freq)
    probs = torch.softmax(logits, dim=1)

ai_prob = probs[0][1].item() * 100
real_prob = probs[0][0].item() * 100

print(f"AI-generated: {ai_prob:.2f}%")
print(f"Real: {real_prob:.2f}%")

