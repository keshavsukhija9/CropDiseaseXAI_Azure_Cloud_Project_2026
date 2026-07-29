import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Biological Knowledge Base for Disease Explanation
DISEASE_KNOWLEDGE_BASE = {
    "Potato___Early_blight": {
        "symptoms": "Dark brown/black spots with concentric rings ('bullseye' pattern) starting on older lower leaves.",
        "cause": "Fungal pathogen Alternaria solani, thriving in high humidity.",
        "xai_focus": "Model attention should highlight concentric ring lesions.",
        "treatment": "Apply copper-based fungicides and practice crop rotation."
    },
    "Tomato___Late_blight": {
        "symptoms": "Large, irregular water-soaked spots turning dark brown.",
        "cause": "Oomycete pathogen Phytophthora infestans.",
        "xai_focus": "Model attention should focus on dark water-soaked leaf margins.",
        "treatment": "Remove infected foliage and apply preventative fungicide."
    }
}

def generate_leaf_explanation(model, input_tensor, raw_rgb_image, target_class_idx, save_path="results/xai_gradcam_output.png"):
    """
    Generates a Grad-CAM heatmap overlay for a leaf image and saves it to results/
    """
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    visualization = show_cam_on_image(raw_rgb_image, grayscale_cam, use_rgb=True)

    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[XAI Pipeline] Grad-CAM visualization saved successfully to: {save_path}")
    
    return grayscale_cam

if __name__ == "__main__":
    print("XAI module initialized. Import `generate_leaf_explanation` into your main training/inference pipeline.")

