import torch
import torch.nn as nn
import torchvision.models as models


class AIDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------------
        # CNN BACKBONE (MobileNetV2)
        # -------------------------------
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Output: 1280 features

        # -------------------------------
        # FREQUENCY BRANCH (COMPRESSED)
        # -------------------------------
        self.freq_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.freq_fc = nn.Sequential(
            nn.Linear(16 * 16, 128),
            nn.ReLU()
        )

        # -------------------------------
        # FINAL CLASSIFIER
        # -------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image, freq):
        # CNN features
        cnn_feat = self.cnn(image)

        # Frequency features
        freq = freq.view(freq.size(0), 1, 224, 224)
        freq = self.freq_pool(freq)
        freq = freq.view(freq.size(0), -1)
        freq_feat = self.freq_fc(freq)

        # Fusion
        fused = torch.cat((cnn_feat, freq_feat), dim=1)

        return self.classifier(fused)

