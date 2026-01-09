


# 🧠 AI vs Real Image Detector

An AI-powered web application that analyzes an uploaded image and predicts whether it is **AI-generated** or **Real**, using deep learning and frequency-based features.

🚀 **Live Demo:**  
👉 https://p-statement-1-d-sol.streamlit.app/

---

## 🏆 Hackathon Context

This project was **built as part of Darshan’s Hackathon**, focusing on solving real-world problems using Artificial Intelligence and Machine Learning.

---

## ✨ Features

- 📤 Upload images (`.jpg`, `.jpeg`, `.png`)
- 🤖 Deep Learning–based classification (AI vs Real)
- 📊 Probability-based confidence scores
- 🧠 FFT-based frequency feature extraction
- ⚡ Fast inference on CPU
- 🌐 Fully deployed using Streamlit Cloud (Free)

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** Streamlit  
- **Backend / ML:** PyTorch, TorchVision  
- **Image Processing:** OpenCV, PIL  
- **Numerical Computing:** NumPy  
- **Model:** Custom CNN (`AIDetector`) trained on AI & real images  

---

## 📂 Project Structure

```

ai_vs_real_detector/
│
├── app.py              # Streamlit entry point
├── model.py            # CNN model definition
├── fft.py              # Frequency-domain feature extraction
├── detector.pth        # Trained model weights
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version config
└── README.md

````

---

## ▶️ How It Works

1. User uploads an image
2. Image is preprocessed and resized
3. Spatial features + FFT frequency features are extracted
4. Features are passed through a trained PyTorch model
5. The app outputs:
   - Probability of **AI-generated**
   - Probability of **Real**
   - Confidence-based interpretation

---

## 🧪 Model Details

- Runs entirely on **CPU**
- Uses **MobileNet-based backbone**
- Trained to detect subtle artifacts common in AI-generated images
- Optimized for lightweight deployment

---

## 🚀 Deployment

The app is deployed for free using **Streamlit Community Cloud**.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
````

---

## 🙌 Acknowledgements

* Built for **Darshan’s Hackathon**
* Thanks to the open-source PyTorch & Streamlit communities

---




