import torch
import numpy as np
import cv2
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from torchvision import transforms

# -----------------------------
# Settings
# -----------------------------
DEVICE = torch.device("cpu")
OPTIMAL_THRESHOLD = 0.2

CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary"]

# -----------------------------
# Load Models
# -----------------------------

# Binary PyTorch model
binary_model = torch.load(
    "models/best_model_full.pth",
    map_location=DEVICE,
    weights_only=False
)
binary_model.eval()

# Multi-class Keras model
multiclass_model = load_model("models/multiclass_tumor_model.h5")

# Same transform used for binary model inference
binary_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Image preprocessing helpers
# -----------------------------

def crop_brain_contour(image, margin=15):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(img_cv, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        img_h, img_w = img_cv.shape

        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(img_w, x + w + margin)
        y_max = min(img_h, y + h + margin)

        image_arr = np.array(image)
        cropped_img = image_arr[y_min:y_max, x_min:x_max]

        return Image.fromarray(cropped_img)

    return image


def normalize_mri_contrast(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=1.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(img_cv)

    return Image.fromarray(
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    )

# -----------------------------
# Prediction Pipeline
# -----------------------------

def predict_pipeline(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    # Preprocessing
    norm_image = normalize_mri_contrast(image)
    cropped_image = crop_brain_contour(norm_image)

    # -----------------------------
    # Stage 1: Binary PyTorch Model
    # -----------------------------
    binary_input = binary_transform(cropped_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        binary_logit = binary_model(binary_input).squeeze(1)
        tumor_probability = torch.sigmoid(binary_logit).cpu().item()

    tumor_probability = float(tumor_probability)

    if tumor_probability < OPTIMAL_THRESHOLD:
        return {
            "prediction": "No Tumor",
            "confidence": 1 - tumor_probability,
            "stage": "Model 1 Binary Filter"
        }

    # -----------------------------
    # Stage 2: Multi-class Keras Model
    # -----------------------------
    img_array = np.array(cropped_image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = preprocess_input(img_array.astype(np.float32))
    img_array = np.expand_dims(img_array, axis=0)

    probs = multiclass_model.predict(img_array, verbose=0).squeeze()

    confidence = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    prediction = CLASS_LABELS[pred_idx]

    # -----------------------------
    # Uncertainty Logic
    # -----------------------------
    if confidence < 0.45:
        return {
            "prediction": "Uncertain (Possibly No Tumor) → Doctor Review",
            "confidence": confidence,
            "stage": "Model 2 Uncertainty OOD Gate",
            "all_probabilities": probs.tolist(),
            "class_labels": CLASS_LABELS
        }

    elif confidence < 0.85:
        return {
            "prediction": f"Uncertain ({prediction} suspected) → Doctor Review",
            "confidence": confidence,
            "stage": "Model 2 Uncertainty Gate",
            "all_probabilities": probs.tolist(),
            "class_labels": CLASS_LABELS
        }

    return {
        "prediction": prediction,
        "confidence": confidence,
        "stage": "Final Tumor Classification",
        "all_probabilities": probs.tolist(),
        "class_labels": CLASS_LABELS
    }