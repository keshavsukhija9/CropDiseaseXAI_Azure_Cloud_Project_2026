# Explainable AI Cloud Platform for Precision Crop Disease Diagnosis using Drone Imagery

## Team Members
- Keshav — 23BIT0262
- Partner — 23BIT0177

## Problem Statement
Crop diseases account for a large share of seasonal yield loss, and manual field scouting cannot keep pace with the large, fragmented holdings typical of Indian agriculture. Existing deep learning models for disease detection are trained and validated on curated, close-up leaf datasets, and their accuracy degrades sharply on real field imagery. A second challenge is trust: most models return only a label and a confidence score, giving an agronomist no way to verify whether the decision was driven by the actual lesion or by background artefacts.

## Objectives
1. Build a cloud pipeline on Azure that ingests drone-captured RGB and multispectral imagery and runs automated crop disease classification on every uploaded field tile.
2. Attach an explainability layer (Grad-CAM / SHAP) so every diagnosis is delivered with a visual justification, not just a label and confidence score.
3. Achieve near-real-time turnaround (under 30 seconds per field tile) using serverless Azure Functions and a managed Azure Machine Learning endpoint.
4. Store geo-tagged diagnosis history and field metadata with role-based access control for farmers, agronomists, and administrators.
5. Push actionable, geo-located alerts (SMS / push notification) with treatment guidance to the field-plot level.
6. Provide a trend dashboard (Power BI) to track disease spread across a season.

## Proposed Architecture/Framework
Two architecture diagrams (full detail in `architecture/`):

**Diagram 1 — Azure Cloud Architecture:** drone/ground station -> Blob Storage -> Data Factory -> Functions -> Azure ML -> XAI service -> Cosmos DB -> API Management -> dashboard and alerts. Microsoft Entra ID handles authentication; Azure Monitor handles observability; VNet and Key Vault secure the data layer.

**Diagram 2 — Complete System Workflow:** field scan -> image acquisition -> cloud ingestion -> AI diagnosis -> explainability -> agronomist review -> farmer alert -> field action, with verified diagnoses routed back for model retraining.

## Technology Stack
- **Cloud Platform:** Microsoft Azure (Blob Storage, Data Factory, Functions, Machine Learning, Cosmos DB, API Management, App Service, Notification Hubs, Entra ID, Monitor, VNet, Key Vault, Power BI)
- **AI/ML:** PyTorch, torchvision, MobileNetV2 (transfer learning), Grad-CAM (pytorch-grad-cam)
- **Backend:** FastAPI, Python 3.12
- **Frontend:** React (Vite)
- **Database:** SQLite (local prototype), migrating to Azure Cosmos DB / Azure SQL

## Dataset Details
- **Dataset Name:** PlantVillage Dataset (color subset)
- **Source:** Kaggle (abdallahalidev/plantvillage-dataset)
- **URL:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Size:** 2.04 GB
- **Records:** 54,305 leaf images
- **Classes:** 38 (across 14 crop species, disease + healthy)
- **License:** CC-BY-NC-SA-4.0
- **Note:** Originally scoped UAV-paired dataset (MH-SoyaHealthVision, 9.75GB) was substituted with PlantVillage to fit the Phase-I timeline.

## Repository Structure
See folder-level README files for the purpose of each directory:
docs/, literature_survey/, architecture/, dataset/, src/frontend/, src/backend/, src/ai_model/, src/azure/, results/, presentation/, references/

See WORK_DISTRIBUTION.md for individual contribution breakdown.
