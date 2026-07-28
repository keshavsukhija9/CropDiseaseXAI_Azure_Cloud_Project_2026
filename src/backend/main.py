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

IMG_SIZE = 224
AI_MODEL_DIR = "../ai_model"
CHECKPOINT_PATH = f"{AI_MODEL_DIR}/checkpoint.pt"
CLASS_MAP_PATH = f"{AI_MODEL_DIR}/class_to_idx.json"
OUTPUT_DIR = "explanations"
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

    explanation_id = str(uuid.uuid4())
    output_path = f"{OUTPUT_DIR}/{explanation_id}.jpg"
    Image.fromarray(visualization).save(output_path)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "explanation_id": explanation_id,
        "explanation_url": f"/explanation/{explanation_id}"
    }


@app.get("/explanation/{explanation_id}")
def get_explanation(explanation_id: str):
    path = f"{OUTPUT_DIR}/{explanation_id}.jpg"
    return FileResponse(path, media_type="image/jpeg")
