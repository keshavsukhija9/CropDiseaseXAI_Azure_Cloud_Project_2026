# Phase-I rubric mapping (BITE412L, Dr. Priya V)

- **Refined title:** "Cross-Scale Explainable AI Cloud Framework for UAV-Based Crop
  Disease Detection, Severity Estimation and Field-Level Decision Support" — used
  verbatim throughout the repo and frontend.
- **Datasets:** MH-SoyaHealthVision UAV (primary, macro, 2,842 images) + MH-SoyaHealthVision
  Leaf (primary, micro, 2,782 images), exact category structure as specified.
  PlantVillage explicitly excluded. SoyNet and India Soybean Dataset reserved for
  external validation (not yet run — future work).
- **Novelty beyond "UAV image -> CNN -> disease":** cross-scale attention fusion
  combining macro (UAV ViT) and micro (leaf CNN) representations
  (`src/ai_model/models/fusion.py`).
- **Severity estimation:** reported as an image-derived disease severity index
  (% affected area + Low/Moderate/High bucket), not expert ground truth, per the
  guidance's methodological caution (`src/ai_model/models/severity.py`).
- **XAI as evaluation, not display:** three methods implemented and run against the
  trained model (Grad-CAM, Grad-CAM++, Integrated Gradients), scored by IoU /
  pointing-game accuracy (`src/ai_model/xai/`). SHAP wrapper coded, not yet
  verified end-to-end (future work).
- **Uncertainty-aware diagnosis:** MC-Dropout entropy drives an auto-accept vs
  agronomist-review decision, live in the `/predict` API response
  (`src/ai_model/models/uncertainty.py`).
- **Robustness under field conditions:** 8-perturbation suite run against the
  trained model with a real accuracy-drop table — blur (-36.7pp) and JPEG
  compression (-21.7pp) are the weakest points (`src/ai_model/robustness/`).
- **Cloud-side drift monitoring:** PSI + KS-test on input/class/confidence
  distributions, exposed via `/drift-status` endpoint
  (`src/ai_model/drift/drift_monitor.py`).
- **System diagram:** `docs/architecture.md`.
- **Working demo:** full pipeline verified live — React frontend -> FastAPI backend
  -> trained fusion model -> Grad-CAM heatmap + severity + uncertainty + decision
  gate, all returned and rendered in a single request.
- **Team split:** Keshav — papers 1-8 gap analysis, backend development, AI/ML
  modeling, database-cloud integration. Chirag — disease data collection,
  frontend/UI, testing.

## Still open before final submission

- Disease-region ground-truth masks for real XAI IoU numbers (currently a
  placeholder mask; pipeline is verified correct, numbers are not meaningful yet).
- SHAP method untested end-to-end (installed, wrapper coded).
- External validation on SoyNet / India Soybean Dataset (per the guidance, these
  are for generalization testing, not training).
- Azure deployment (next phase — App Service F1, Blob, Cosmos DB free tier,
  Static Web Apps, Application Insights — all free-tier only).
