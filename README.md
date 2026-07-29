# CropDiseaseXAI: Explainable AI for Plant Pathology on Azure Cloud 🌾🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)
[![Cloud](https://img.shields.io/badge/Azure-Cloud_Integrated-0078D4.svg)](https://azure.microsoft.com/)

An end-to-end computer vision and Explainable AI (XAI) framework designed to classify crop leaf diseases and generate interpretable Grad-CAM visual heatmaps alongside biological symptom context.

---

## 📌 Features & Core Highlights
- **Computer Vision Core:** High-accuracy disease classification using deep transfer learning.
- **Explainable AI (XAI):** Visual feature attribution via Grad-CAM heatmaps overlaying key leaf lesions.
- **Biological Knowledge Base:** Automatic mapping of predicted diseases to clinical symptoms, causes, and treatments.
- **Cloud Readiness:** Architected for deployment and dataset storage on Azure Cloud infrastructure.
- **Clean Repository Setup:** Modular layout with `.gitignore` and `.gitkeep` dataset structure.

---

## 📂 Project Architecture

```text
CropDiseaseXAI_Azure_Cloud_Project_2026/
├── architecture/      # System workflow diagrams & cloud infrastructure design
├── dataset/           # Local data root (git-ignored, structure kept via .gitkeep)
├── docs/              # Research papers, literature survey, and Phase-I reports
├── literature_survey/ # Comparative analysis of existing crop models
├── presentation/      # Project presentation decks (.pptx / .pdf)
├── references/        # Citation references & papers
├── results/           # Saved Grad-CAM heatmaps, evaluation metrics, and plots
├── src/               # Core codebase
│   └── explain.py     # Grad-CAM XAI explanation module & disease knowledge base
├── .gitignore         # Excludes heavy datasets and model checkpoints
├── requirements.txt   # Core Python dependencies
└── README.md          # Project documentation# Explainable AI Cloud Platform for Precision Crop Disease Diagnosis using Drone Imagery
