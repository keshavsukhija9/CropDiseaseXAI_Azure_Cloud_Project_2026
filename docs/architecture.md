# Architecture

## System overview

```
                    Leaf image (micro)         React frontend
                    UAV image (macro)          (upload + results)
                           |                           |
                           v                           |
                    FastAPI backend (/predict)  <-------
                           |
              +------------+------------+
              v                         v
     Leaf CNN (MobileNetV2)   UAV ViT (vit_small_patch16_224)
              |                         |
              +---> Cross-Scale Attention Fusion <---+
                           |
          +----------------+-----------------+
          v                v                 v
   Disease classifier  Severity head    MC-Dropout uncertainty
   (6 classes)         (image-derived   (entropy over 20 passes)
                        % affected,
                        Low/Mod/High)
          |                |                 |
          +----------------+--------+--------+
                                    v
                     Decision gate: low uncertainty -> auto-accept
                                    high uncertainty -> agronomist review
                                    |
                                    v
                            SQLite (prediction log + history)
                                    |
                                    v
                        /drift-status: PSI + KS-test on rolling
                        prediction window (Module 6)
```

## XAI evaluation flow (Module 3)

```
Leaf image -> [Grad-CAM | Grad-CAM++ | Integrated Gradients] -> saliency map
                                                                      |
                                                   (disease-region ground truth
                                                    not yet available; placeholder
                                                    mask used for pipeline testing)
                                                                      v
                                              IoU + pointing-game accuracy per method
```

Verified output on a real trained sample (class: Healthy):
- Grad-CAM: IoU 0.188, pointing-game miss
- Grad-CAM++: IoU 0.399, pointing-game miss
- Integrated Gradients: IoU 0.215, pointing-game hit

(Numbers are against a placeholder 80x80 center mask, not real disease-region
ground truth -- included to demonstrate the evaluation pipeline runs correctly,
not as a claim about localization quality. Real ground-truth masks are future work.)

## Robustness results (Module 5, n=60 validation images)

| Condition | Accuracy |
|---|---|
| Clean | 0.700 |
| Rotation | 0.733 |
| Shadow | 0.650 |
| Occlusion | 0.650 |
| Color shift | 0.617 |
| Haze | 0.567 |
| Brightness | 0.550 |
| JPEG compression | 0.483 |
| Blur | 0.333 |

Blur and JPEG compression cause the largest accuracy drops -- motivates
blur/compression-aware augmentation as future work, exactly the experimental
contribution the guidance document asks for.

## Model performance (Module 1, full dataset, dropout off)

Trained on MH-SoyaHealthVision (2,782 leaf + 2,842 UAV images, 6 leaf classes),
15 epochs, MobileNetV2 + ViT-small fusion, Apple M4 (MPS).

- Best validation accuracy: 0.890 (held-out 15% split)
- Full-dataset accuracy: 0.964, macro-F1: 0.953

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 1.000 | 1.000 | 1.000 |
| Mosaic | 0.968 | 0.989 | 0.978 |
| Rust | 0.968 | 0.955 | 0.962 |
| Septoria brown spot | 0.930 | 0.896 | 0.913 |
| Frogeye leaf spot | 0.892 | 0.882 | 0.887 |
| Caterpillar/Semi-looper | 0.975 | 0.988 | 0.981 |

## Azure resource map (free tier only)

| Component | Service |
|---|---|
| Inference API | App Service Linux, F1 (free) plan |
| Image storage | Blob Storage |
| Prediction log / feedback loop | Cosmos DB free tier |
| Dashboard | Static Web Apps (Free) |
| CI/CD | GitHub Actions |
| Monitoring | Application Insights (free quota) |
