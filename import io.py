import io
import base64
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageChops

# -----------------------------
# Config
# -----------------------------
PATCH_SIZE = 256
PATCH_OVERLAP = 0.25
RESIZE_TARGET = 1024
JPEG_QUALITY = 60
ABSTAIN_TAU = 0.25  # uncertainty threshold

# -----------------------------
# Preprocessing & multi-view
# -----------------------------
def center_crop_resize(img, target=RESIZE_TARGET):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((target, target))
    return img

def jpeg_recompress(img, quality=JPEG_QUALITY):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

def multi_view_variants(img):
    # Original, recompressed, resized (0.75x)
    original = img
    recompressed = jpeg_recompress(img)
    resized = img.resize((max(64, int(img.size[0]*0.75)), max(64, int(img.size[1]*0.75))))
    return [original, recompressed, resized]

def make_patches(img, size=PATCH_SIZE, overlap=PATCH_OVERLAP):
    step = max(1, int(size * (1 - overlap)))
    patches = []
    for y in range(0, img.size[1] - size + 1, step):
        for x in range(0, img.size[0] - size + 1, step):
            patches.append(img.crop((x, y, x + size, y + size)))
    return patches

# -----------------------------
# Branches (stable placeholders)
# -----------------------------
class PixelBranch:
    def extract(self, patch: Image.Image):
        # Noise residual via median filter difference + simple stats
        gray = patch.convert("L")
        smoothed = gray.filter(ImageFilter.MedianFilter(size=3))
        residual = ImageChops.difference(gray, smoothed)
        residual = ImageOps.autocontrast(residual)
        arr_g = np.array(gray, dtype=np.float32)
        arr_r = np.array(residual, dtype=np.float32)
        feats = [
            arr_g.mean(), arr_g.std(),
            float(np.mean(np.abs(np.diff(arr_g, axis=0)))),
            float(np.mean(np.abs(np.diff(arr_g, axis=1)))),
            arr_r.mean(), arr_r.std()
        ]
        return np.pad(np.array(feats, dtype=np.float32), (0, 122), mode='constant')

class FrequencyBranch:
    def extract(self, patch: Image.Image):
        arr = np.array(patch.convert("L"), dtype=np.float32)
        # FFT magnitude spectrum (robust to small images)
        fft = np.fft.fft2(arr)
        mag = np.abs(np.fft.fftshift(fft))
        h, w = mag.shape
        cy, cx = h//2, w//2
        # Radial energy profile (coarse rings)
        rings = []
        for r in [8, 16, 32, 64, 96]:
            y, x = np.ogrid[:h, :w]
            mask = (y - cy)**2 + (x - cx)**2 <= r**2
            rings.append(float(mag[mask].mean()))
        feats = np.array(rings, dtype=np.float32)
        return np.pad(feats, (0, 123), mode='constant')

class SemanticBranch:
    def extract(self, patch: Image.Image):
        # Texture/edge stats as a proxy for semantic oddities
        edges = patch.convert("L").filter(ImageFilter.FIND_EDGES)
        earr = np.array(edges, dtype=np.float32)
        # Color channel stats
        carr = np.array(patch, dtype=np.float32)
        feats = [
            earr.mean(), earr.std(),
            float(np.percentile(earr, 90)),
            float(np.percentile(earr, 99)),
            carr[...,0].mean(), carr[...,1].mean(), carr[...,2].mean(),
            carr[...,0].std(), carr[...,1].std(), carr[...,2].std()
        ]
        return np.pad(np.array(feats, dtype=np.float32), (0, 246), mode='constant')

# -----------------------------
# Fusion & calibration (placeholders)
# -----------------------------
class FusionClassifier:
    def forward(self, f_pixel, f_freq, f_sem):
        x = np.concatenate([f_pixel, f_freq, f_sem])
        # Normalize to avoid extreme values
        xm = x.mean()
        xs = np.std(x) + 1e-6
        z = xm / xs
        p_ai = 1 / (1 + np.exp(-z))
        return np.array([p_ai, 1 - p_ai], dtype=np.float32)

def temperature_scale(p, temperature=1.2):
    # Apply temperature scaling to logits proxy
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p[:,0]) - np.log(p[:,1])
    scaled = 1 / (1 + np.exp(-logit / temperature))
    out = p.copy()
    out[:,0] = scaled
    out[:,1] = 1 - scaled
    return out

def ensemble_predict(patch_probs_list):
    # Treat patch_probs_list as probabilities from multiple patches/views
    probs = np.array(patch_probs_list)  # shape [N, 2]
    var = float(np.var(probs[:,0]))
    energy = float(-np.log(np.mean(probs[:,0]*(1-probs[:,0]) + 1e-6)))
    return probs, var, energy

def abstain_logic(p_ai, uncertainty, tau=ABSTAIN_TAU):
    if uncertainty > tau:
        return "Inconclusive", "High uncertainty—transformations or content obscure forensic cues."
    return ("AI" if p_ai >= 0.5 else "Real"), "Confidence based on multi-view, multi-patch consensus."

# -----------------------------
# Explainability (placeholders)
# -----------------------------
def to_base64_image(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_saliency(img: Image.Image):
    # Proxy saliency: edges + contrast
    sal = img.convert("L").filter(ImageFilter.FIND_EDGES)
    sal = ImageOps.autocontrast(sal)
    return to_base64_image(sal)

def generate_frequency_heatmap(img: Image.Image):
    # Proxy frequency emphasis: unsharp mask
    freq = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return to_base64_image(freq)

# -----------------------------
# Utility: safe open
# -----------------------------
def safe_open_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI vs Real Image Detector", layout="centered")
st.title("AI-Generated vs Real Image Detector")

st.write("Upload an image to analyze whether it's AI-generated or a real photograph. Outputs include confidence, uncertainty, and visual explanations.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
if uploaded:
    img = safe_open_image(uploaded.getvalue())
    if img is None:
        st.error("Could not read the image. Please upload a valid PNG/JPG/JPEG/WEBP file.")
    else:
        st.image(img, caption="Input", use_column_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                # Initialize branches and fusion
                pixel = PixelBranch()
                freq = FrequencyBranch()
                sem = SemanticBranch()
                fusion = FusionClassifier()

                # Multi-view preprocessing
                views = multi_view_variants(img)
                patch_probs = []

                for v in views:
                    pre = center_crop_resize(v)
                    patches = make_patches(pre, size=PATCH_SIZE, overlap=PATCH_OVERLAP)
                    # Guard against tiny images producing zero patches
                    if not patches:
                        patches = [pre.resize((PATCH_SIZE, PATCH_SIZE))]
                    for p in patches:
                        f_pixel = pixel.extract(p)
                        f_freq  = freq.extract(p)
                        f_sem   = sem.extract(p)
                        probs = fusion.forward(f_pixel, f_freq, f_sem)
                        patch_probs.append(probs)

                # Ensemble + calibration
                probs, var, energy = ensemble_predict(patch_probs)
                probs_cal = temperature_scale(probs, temperature=1.2)

                # Aggregate
                p_ai = float(np.mean(probs_cal[:,0]))
                p_real = 1.0 - p_ai
                uncertainty = float(var + energy)

                decision, reason = abstain_logic(p_ai, uncertainty, tau=ABSTAIN_TAU)

                # Explanations (central view)
                central = center_crop_resize(views[0])
                sal_b64 = generate_saliency(central)
                freq_b64 = generate_frequency_heatmap(central)

            # Display results
            st.subheader("Prediction")
            col1, col2, col3 = st.columns(3)
            col1.metric("p(AI)", f"{p_ai:.3f}")
            col2.metric("p(Real)", f"{p_real:.3f}")
            col3.metric("Uncertainty", f"{uncertainty:.3f}")
            st.write(f"Decision: **{decision}**")
            st.caption(reason)

            st.subheader("Explanations")
            st.image(base64.b64decode(sal_b64), caption="Saliency (regions influencing decision)")
            st.image(base64.b64decode(freq_b64), caption="Frequency heatmap (artifact emphasis)")