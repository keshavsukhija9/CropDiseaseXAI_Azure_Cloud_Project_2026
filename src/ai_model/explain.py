import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import json
import sys

IMG_SIZE = 224
CHECKPOINT_PATH = "checkpoint.pt"
CLASS_MAP_PATH = "class_to_idx.json"

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

def predict_and_explain(image_path, output_path="explanation.jpg"):
    img = Image.open(image_path).convert("RGB")
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
    Image.fromarray(visualization).save(output_path)

    return predicted_class, confidence, output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 explain.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    predicted_class, confidence, output_path = predict_and_explain(image_path)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Heatmap saved to: {output_path}")
