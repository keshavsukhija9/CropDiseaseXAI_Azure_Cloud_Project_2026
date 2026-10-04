import io
import os
import json
import uuid
import sqlite3
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.ai_model.config import LEAF_CLASSES, IMAGE_SIZE
from src.ai_model.data.datasets import default_transform
from src.ai_model.models.full_model import CrossScaleDiseaseModel
from src.ai_model.models.severity import SeverityHead
from src.ai_model.models.uncertainty import mc_dropout_predict, decide
from src.ai_model.xai.gradcam import run_gradcam

CONFIG_PATH = os.getenv("APP_CONFIG", "configs/default.yaml")
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "runs/best.pt")
OUTPUT_DIR = "explanations"
DB_PATH = "history.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

app = FastAPI(title="Cross-Scale Crop Disease XAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CrossScaleDiseaseModel(
    num_classes=CFG["model"]["num_classes"],
    leaf_backbone=CFG["model"]["leaf_backbone"],
    uav_backbone=CFG["model"]["uav_backbone"],
    fusion_dim=CFG["model"]["fusion_dim"],
    pretrained=False,
).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

transform = default_transform(train=False)



def get_treatment_advice(class_name):
    name = class_name.lower()
    if "healthy" in name:
        return "No treatment needed. Continue routine monitoring."
    if "rust" in name:
        return "Apply a rust-specific fungicide (e.g. myclobutanil or propiconazole). Remove severely infected leaves and improve field airflow."
    if "septoria" in name:
        return "Remove and destroy affected lower leaves. Apply a strobilurin or triazole fungicide. Rotate crops to reduce inoculum carryover."
    if "frogeye" in name:
        return "Apply a QoI or triazole fungicide at early infection. Use resistant soybean varieties in future plantings."
    if "mosaic" in name:
        return "No chemical cure. Remove and destroy infected plants. Control aphid vectors with insecticidal soap or approved insecticide."
    if "caterpillar" in name or "semi_looper" in name or "looper" in name:
        return "Apply Bt (Bacillus thuringiensis) spray or an approved insecticide. Scout field edges regularly during pest season."
    return "Consult a local agronomist for a targeted treatment plan."


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            leaf_filename TEXT,
            uav_filename TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity_pct REAL NOT NULL,
            severity_bucket TEXT NOT NULL,
            uncertainty REAL NOT NULL,
            decision TEXT NOT NULL,
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
async def predict(leaf_image: UploadFile = File(...), uav_image: UploadFile = File(...)):
    leaf_bytes = await leaf_image.read()
    uav_bytes = await uav_image.read()

    leaf_img = Image.open(io.BytesIO(leaf_bytes)).convert("RGB")
    uav_img = Image.open(io.BytesIO(uav_bytes)).convert("RGB")

    leaf_tensor = transform(leaf_img).unsqueeze(0).to(device)
    uav_tensor = transform(uav_img).unsqueeze(0).to(device)

    # deterministic prediction (dropout off) for the headline class + severity
    with torch.no_grad():
        logits, severity_frac, _ = model(leaf_tensor, uav_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()

    predicted_class = LEAF_CLASSES[pred_idx]
    severity_pct = round(float(severity_frac.item()) * 100, 1)
    severity_bucket = SeverityHead.bucket(
        severity_pct, CFG["severity"]["low_max_pct"], CFG["severity"]["moderate_max_pct"]
    )

    # Module 4: MC-Dropout uncertainty -> decision gate
    _, norm_entropy = mc_dropout_predict(model, leaf_tensor, uav_tensor, passes=CFG["model"]["mc_dropout_passes"])
    model.eval()  # mc_dropout_predict flips dropout to train mode; reset
    uncertainty = round(float(norm_entropy.item()), 3)
    decision = decide(
        uncertainty, CFG["uncertainty"]["entropy_low_threshold"], CFG["uncertainty"]["entropy_high_threshold"]
    )

    # Module 3: Grad-CAM visualization saved for the frontend
    rgb_leaf = np.array(leaf_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    grayscale_cam = run_gradcam(model, leaf_tensor, uav_tensor, class_idx=pred_idx, plus_plus=False)[0]
    visualization = show_cam_on_image(rgb_leaf.astype(np.float32), grayscale_cam, use_rgb=True)

    treatment = get_treatment_advice(predicted_class)

    explanation_id = str(uuid.uuid4())
    Image.fromarray(visualization).save(f"{OUTPUT_DIR}/{explanation_id}.jpg")

    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (id, timestamp, leaf_filename, uav_filename, predicted_class, confidence,
            severity_pct, severity_bucket, uncertainty, decision, explanation_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, timestamp, leaf_image.filename, uav_image.filename, predicted_class,
         confidence, severity_pct, severity_bucket, uncertainty, decision, explanation_id),
    )
    conn.commit()
    conn.close()

    return {
        "id": prediction_id,
        "timestamp": timestamp,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "severity_pct": severity_pct,
        "severity_bucket": severity_bucket,
        "uncertainty": uncertainty,
        "decision": decision,
        "treatment": treatment,
        "explanation_id": explanation_id,
        "explanation_url": f"/explanation/{explanation_id}",
    }


@app.get("/explanation/{explanation_id}")
def get_explanation(explanation_id: str):
    return FileResponse(f"{OUTPUT_DIR}/{explanation_id}.jpg", media_type="image/jpeg")


@app.get("/history")
def get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, timestamp, leaf_filename, uav_filename, predicted_class, confidence,
                  severity_pct, severity_bucket, uncertainty, decision, explanation_id
           FROM predictions ORDER BY timestamp DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/drift-status")
def drift_status():
    """
    Module 6: drift check against logged predictions. Compares the most
    recent 50 predictions' confidence distribution against the prior 50
    (reference window) using PSI + KS-test.
    """
    from src.ai_model.drift.drift_monitor import check_drift

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT confidence, predicted_class FROM predictions ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    conn.close()

    if len(rows) < 20:
        return {"status": "insufficient_data", "message": "Need at least 20 logged predictions to assess drift."}

    confidences = np.array([r[0] for r in rows])
    classes = np.array([LEAF_CLASSES.index(r[1]) for r in rows])
    mid = len(rows) // 2
    current_conf, reference_conf = confidences[:mid], confidences[mid:]
    current_cls, reference_cls = classes[:mid], classes[mid:]

    report = check_drift(reference_conf, current_conf, reference_cls, current_cls, reference_conf, current_conf)
    return {
        "drift_detected": report.drift_detected,
        "reasons": report.reasons,
        "input_psi": round(report.input_psi, 4),
        "class_dist_ks_pvalue": round(report.class_dist_ks_pvalue, 4),
        "confidence_psi": round(report.confidence_psi, 4),
    }
