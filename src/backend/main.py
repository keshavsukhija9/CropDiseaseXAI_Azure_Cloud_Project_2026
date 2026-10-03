from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import json
import io
import uuid
import os
import sqlite3
from datetime import datetime, timezone

IMG_SIZE = 224
AI_MODEL_DIR = "../ai_model"
CHECKPOINT_PATH = f"{AI_MODEL_DIR}/checkpoint.pt"
CLASS_MAP_PATH = f"{AI_MODEL_DIR}/class_to_idx.json"
OUTPUT_DIR = "explanations"
DB_PATH = "history.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Crop Disease XAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open(CLASS_MAP_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_treatment_advice(class_name):
    name = class_name.lower()
    if "healthy" in name:
        return "No treatment needed. Continue routine monitoring."
    if "blight" in name:
        return "Remove and destroy infected leaves. Apply a copper-based or chlorothalonil fungicide. Improve air circulation and avoid overhead watering."
    if "rust" in name:
        return "Prune affected areas. Apply a rust-specific fungicide (e.g. myclobutanil). Remove nearby alternate host plants if applicable (e.g. cedar for apple rust)."
    if "mildew" in name:
        return "Apply sulfur or potassium bicarbonate-based fungicide. Increase airflow between plants and avoid wetting foliage."
    if "spot" in name:
        return "Remove affected leaves. Apply a broad-spectrum fungicide. Avoid working in the field when foliage is wet."
    if "rot" in name or "scab" in name:
        return "Remove and dispose of infected plant material away from the field. Apply an appropriate fungicide during the growing season."
    if "mosaic" in name or "virus" in name or "curl" in name:
        return "No chemical cure. Remove and destroy infected plants to prevent spread. Control insect vectors (aphids/whiteflies) with insecticidal soap."
    if "mite" in name:
        return "Apply miticide or insecticidal soap. Increase humidity around plants, as spider mites thrive in dry conditions."
    if "greening" in name or "huanglongbing" in name:
        return "No cure available. Remove and destroy infected trees to prevent spread. Control psyllid insect vectors."
    if "bacterial" in name:
        return "Apply copper-based bactericide. Avoid overhead irrigation and remove infected plant debris."
    return "Consult a local agronomist for a targeted treatment plan."


def compute_severity(grayscale_cam, threshold=0.5):
    activated_fraction = float((grayscale_cam >= threshold).mean())
    return round(activated_fraction * 100, 1)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            filename TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity_pct REAL NOT NULL,
            explanation_id TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_img = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    predicted_class = idx_to_class[pred_idx]

    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    severity_pct = compute_severity(grayscale_cam)
    treatment = get_treatment_advice(predicted_class)

    explanation_id = str(uuid.uuid4())
    output_path = f"{OUTPUT_DIR}/{explanation_id}.jpg"
    Image.fromarray(visualization).save(output_path)

    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (id, timestamp, filename, predicted_class, confidence, severity_pct, explanation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (prediction_id, timestamp, file.filename, predicted_class, confidence, severity_pct, explanation_id),
    )
    conn.commit()
    conn.close()

    return {
        "id": prediction_id,
        "timestamp": timestamp,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "severity_pct": severity_pct,
        "treatment": treatment,
        "explanation_id": explanation_id,
        "explanation_url": f"/explanation/{explanation_id}",
    }


@app.get("/explanation/{explanation_id}")
def get_explanation(explanation_id: str):
    path = f"{OUTPUT_DIR}/{explanation_id}.jpg"
    return FileResponse(path, media_type="image/jpeg")


@app.get("/history")
def get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, timestamp, filename, predicted_class, confidence, severity_pct, explanation_id FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
