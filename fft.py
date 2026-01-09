import torch

def fft_features(img):
    img = img.mean(dim=1)
    fft = torch.fft.fft2(img)
    fft = torch.abs(fft)
    return fft.view(fft.size(0), -1)
