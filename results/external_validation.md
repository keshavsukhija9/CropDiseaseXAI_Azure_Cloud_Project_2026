# External validation — CrossScaleDiseaseModel

Generated 2026-10-03 · checkpoint `runs/best.pt` · config `configs/default.yaml` · device `cpu`

Binary task: **Healthy vs. Diseased**. The model's 5 disease classes (Mosaic, Rust, Septoria_brown_spot, Frogeye_leaf_spot, Caterpillar_Semi_looper) are collapsed to *Diseased*; *Healthy* stays *Healthy*. Precision, recall and F1 treat **Diseased as the positive class**.

> **UAV input:** these datasets have no UAV images. Every leaf image was paired with a neutral UAV input (an all-zero tensor after normalization). See *Limitations*.

## Image counts per folder

| Dataset | Folder | Split | Images |
|---|---|---|---|
| SoyNet | Disease_Pic | all | 2762 |
| SoyNet | Healthy_pic | all | 445 |
| India Soybean | 1.Healthy | Multi | 29 |
| India Soybean | 1.Healthy | Single | 259 |
| India Soybean | 2.Vein Necrosis | Multi | 34 |
| India Soybean | 2.Vein Necrosis | Single | 104 |
| India Soybean | 3.Dry_leaf | Multi | 31 |
| India Soybean | 3.Dry_leaf | Single | 199 |
| India Soybean | 4.Septoria_Brown_Spot | Multi | 42 |
| India Soybean | 4.Septoria_Brown_Spot | Single | 242 |
| India Soybean | 6.Bacterial leaf Blight | unsplit | 226 |

SoyNet's mobile-phone images were **excluded** because they have no labels; only the labelled *Camera Clicks* set is used.

## Main result — Healthy vs. Diseased

| Dataset | Split | N | True Healthy | True Diseased | Accuracy | Balanced acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| SoyNet | all | 3207 | 445 | 2762 | 0.861 | 0.528 | 0.868 | 0.988 | 0.924 |
| India Soybean | Single | 605 | 259 | 346 | 0.577 | 0.506 | 0.575 | 1.000 | 0.730 |
| India Soybean | Multi | 105 | 29 | 76 | 0.724 | 0.500 | 0.724 | 1.000 | 0.840 |
| India Soybean | Combined | 936 | 288 | 648 | 0.696 | 0.505 | 0.695 | 1.000 | 0.820 |

India Soybean excludes *3.Dry_leaf* (reported separately). *6.Bacterial leaf Blight* has no Single/Multi split, so it only appears in the **Combined** row.

### Confusion matrices (rows = true, columns = predicted)

**SoyNet — all** (n = 3207)

|  | Pred Healthy | Pred Diseased |
|---|---|---|
| True Healthy | 30 | 415 |
| True Diseased | 32 | 2730 |

**India Soybean — Single** (n = 605)

|  | Pred Healthy | Pred Diseased |
|---|---|---|
| True Healthy | 3 | 256 |
| True Diseased | 0 | 346 |

**India Soybean — Multi** (n = 105)

|  | Pred Healthy | Pred Diseased |
|---|---|---|
| True Healthy | 0 | 29 |
| True Diseased | 0 | 76 |

**India Soybean — Combined** (n = 936)

|  | Pred Healthy | Pred Diseased |
|---|---|---|
| True Healthy | 3 | 285 |
| True Diseased | 0 | 648 |

## India Soybean — Dry_leaf (not in the main result)

| Split | N | Predicted Healthy | Predicted Diseased |
|---|---|---|---|
| Single | 199 | 0.0% | 100.0% |
| Multi | 31 | 0.0% | 100.0% |
| Combined | 230 | 0.0% | 100.0% |

Dry leaf may be senescence or drought stress rather than a disease, so there is no correct binary label for it. These numbers show how the model behaves on it, not whether it is right.

## Exact-class check (6-class)

| Folder | Split | N | Expected class | Exact hits | Exact % |
|---|---|---|---|---|---|
| 1.Healthy | Single | 259 | Healthy | 3 | 1.2% |
| 1.Healthy | Multi | 29 | Healthy | 0 | 0.0% |
| 1.Healthy | Combined | 288 | Healthy | 3 | 1.0% |
| 4.Septoria_Brown_Spot | Single | 242 | Septoria_brown_spot | 114 | 47.1% |
| 4.Septoria_Brown_Spot | Multi | 42 | Septoria_brown_spot | 39 | 92.9% |
| 4.Septoria_Brown_Spot | Combined | 284 | Septoria_brown_spot | 153 | 53.9% |

## Predicted 6-class distribution per folder

| Dataset | Folder | Split | N | Healthy | Mosaic | Rust | Septoria_brown_spot | Frogeye_leaf_spot | Caterpillar_Semi_looper |
|---|---|---|---|---|---|---|---|---|---|
| SoyNet | Disease_Pic | all | 2762 | 32 | 2057 | 10 | 639 | 24 | 0 |
| SoyNet | Healthy_pic | all | 445 | 30 | 173 | 0 | 242 | 0 | 0 |
| India Soybean | 1.Healthy | Multi | 29 | 0 | 29 | 0 | 0 | 0 | 0 |
| India Soybean | 1.Healthy | Single | 259 | 3 | 207 | 0 | 49 | 0 | 0 |
| India Soybean | 2.Vein Necrosis | Multi | 34 | 0 | 29 | 0 | 5 | 0 | 0 |
| India Soybean | 2.Vein Necrosis | Single | 104 | 0 | 104 | 0 | 0 | 0 | 0 |
| India Soybean | 3.Dry_leaf | Multi | 31 | 0 | 31 | 0 | 0 | 0 | 0 |
| India Soybean | 3.Dry_leaf | Single | 199 | 0 | 193 | 6 | 0 | 0 | 0 |
| India Soybean | 4.Septoria_Brown_Spot | Multi | 42 | 0 | 3 | 0 | 39 | 0 | 0 |
| India Soybean | 4.Septoria_Brown_Spot | Single | 242 | 0 | 121 | 7 | 114 | 0 | 0 |
| India Soybean | 6.Bacterial leaf Blight | unsplit | 226 | 0 | 216 | 0 | 10 | 0 | 0 |

## Limitations

- **No UAV input.** The model is a leaf + UAV fusion network trained with real UAV images. Here the UAV branch received an all-zero tensor after normalization (equivalent to a flat image of the ImageNet mean colour), which the model never saw in training. These results measure the leaf pathway under an out-of-distribution UAV input, **not** the full system as designed, and may understate or misrepresent its performance.
- **SoyNet mobile images excluded.** They have no labels, so only the labelled Camera Clicks photos were used.
- **Binary collapse.** The external label sets don't match our 6 classes, so most of the evaluation is Healthy vs. Diseased only. A "Diseased" hit can come from the wrong disease class; the exact-class check covers only Healthy and Septoria brown spot.
- **Diseases outside our label set.** Vein necrosis and bacterial leaf blight are not among our trained classes; counting them as correct whenever the model predicts any disease is a generous assumption.
- **Dry_leaf has no ground truth** for this task and is kept out of the main metrics.
- **6.Bacterial leaf Blight has no Single/Multi split**, so Single and Multi results for India exclude it and are not directly comparable to Combined.
- **Class imbalance.** SoyNet is about 6:1 Diseased to Healthy and India about 2:1, so plain accuracy is inflated by always predicting Diseased; read balanced accuracy alongside it.
- **Preprocessing.** Full-size photos are resized straight to 224×224 like in training (no crop, aspect ratio not preserved, EXIF orientation not applied).
- **Domain shift.** Different cameras, backgrounds, lighting and regions from the training data; single runs with no confidence intervals.
