import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import AIDetector
from fft import fft_features

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data/", transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cpu")
model = AIDetector().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        freq = fft_features(img).to(device)

        out = model(img, freq)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "detector.pth")

