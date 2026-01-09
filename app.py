import numpy as np
import streamlit as st
import torch
import cv2
from PIL import Image
from torchvision import transforms
from model import AIDetector
from fft import fft_features

# Page config
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI vs Real Image Detector")
st.write("Upload an image to check whether it is **AI-generated or Real**.")

# Device
device = torch.device("cpu")

# Load model
@st.cache_resource
def load_model():
    model = AIDetector().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "detector.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Convert for model
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # FFT features
    freq = fft_features(img_tensor).to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor, freq)
        probs = torch.softmax(logits, dim=1)

    ai_prob = probs[0][1].item() * 100
    real_prob = probs[0][0].item() * 100

    st.markdown("## 🔍 Prediction Result")
    st.progress(ai_prob / 100)

    st.write(f"🤖 **AI-generated:** `{ai_prob:.2f}%`")
    st.write(f"📷 **Real:** `{real_prob:.2f}%`")

    # Confidence interpretation
    if ai_prob > 70:
        st.success("High confidence: AI-generated image")
    elif ai_prob < 30:
        st.success("High confidence: Real image")
    else:
        st.warning("⚠️ Uncertain prediction (ambiguous case)")