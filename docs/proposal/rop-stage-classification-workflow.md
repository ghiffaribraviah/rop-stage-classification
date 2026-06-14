# Proposed Workflow: ROP Stage Classification

## 1. Project Objective

This project builds an image classification system for Retinopathy of Prematurity (ROP) stage classification using the Zhao2024 dataset. The main classification target is:

- Normal
- Stage 1 ROP
- Stage 2 ROP
- Stage 3 ROP

The project compares four input scenarios to analyze how image enhancement and vessel segmentation affect classification performance.

## 2. Assignment Alignment

The assignment requires a complete image-analysis pipeline, not only a training result. This workflow covers the required components:

- Dataset acquisition and understanding
- Data preprocessing
- Image enhancement or restoration
- Segmentation of important structures
- Feature extraction
- Classification
- Performance evaluation
- Critical analysis of strengths, weaknesses, and future improvements

## 3. Main References

The project uses the following references from `docs/references`:

- `Zhao2024.pdf`: main classification dataset reference.
- `Rahim2024.pdf`: image enhancement reference for ROP stage classification. Its dataset is different from Zhao2024, so the enhancement is treated as an experimental transfer idea, not as a guaranteed improvement.
- `Almeida2024.pdf`: vessel enhancement and segmentation reference.
- `Agrawal2021.pdf`: extra dataset and reference for evaluating vessel segmentation quality.
- `Vahidmoghadam2026.pdf`: scenario inspiration, especially the combined image plus vessel mask input.

## 4. Dataset Plan

### 4.1 Main Dataset

Use the Zhao2024 image dataset in `data/image`.

Main classes:

| Class | Image Count |
|---|---:|
| Normal | 236 |
| Stage 1 | 94 |
| Stage 2 | 165 |
| Stage 3 | 261 |
| Total | 756 |

The `laser scars` class is excluded from the main classification task because it represents post-treatment appearance, not a primary ROP stage. It can be discussed as a limitation or future extension.

### 4.2 Segmentation Evaluation Dataset

Use the HVDROPDB segmentation data in `data/HVDROPDB_RetCam_Neo_Segmentation` as an extra dataset for vessel segmentation evaluation.

Relevant folders:

- `HVDROPDB-BV`: blood vessel images and masks
- `HVDROPDB-OD`: optic disc images and masks
- `HVDROPDB-RIDGE`: ridge images and masks

The primary use in this project is vessel segmentation quality evaluation using the available vessel masks.

## 5. Dataset Split

Use the same split for all classification scenarios to keep the comparison fair.

Follow the published Zhao2024 split ratio:

- Training: 80%
- Validation: 10%
- Test: 10%

For this project's 4-class setup, excluding `laser scars`, the expected split is:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Normal | 188 | 24 | 24 | 236 |
| Stage 1 | 75 | 9 | 10 | 94 |
| Stage 2 | 132 | 16 | 17 | 165 |
| Stage 3 | 208 | 26 | 27 | 261 |
| Total | 603 | 75 | 78 | 756 |

Use stratified splitting to preserve this class distribution as closely as possible.

Zhao2024 reports that the full dataset contains 1,099 images from 789 eyes and 483 infants. However, the local dataset filenames do not expose infant ID or eye ID, and the paper does not clearly state that the published split is patient-level or eye-level grouped. Because of this, a standard image-level split may introduce leakage if images from the same infant or eye appear in different splits.

If patient-level or eye-level metadata becomes available, use group-stratified splitting by infant ID or eye ID. If metadata is unavailable, use stratified image-level splitting with a fixed random seed and explicitly state this limitation in the report.

Suggested report wording:

```text
Because patient-level identifiers were not available in the local dataset files, this study used stratified image-level splitting following the Zhao2024 8:1:1 split ratio. This may introduce data leakage if multiple images from the same infant or eye appear across training and test sets. Therefore, the reported performance should be interpreted as image-level performance rather than fully patient-independent generalization.
```

## 6. Shared Preprocessing

Apply the following shared preprocessing before scenario-specific processing:

1. Load image and label.
2. Remove excluded classes, especially `laser scars`.
3. Resize image to a fixed input size, for example `224x224` or `256x256`.
4. Normalize pixel values to `[0, 1]` or use model-specific normalization.
5. Apply mild augmentation only to training data:
   - Horizontal flip
   - Vertical flip if clinically acceptable
   - Small rotation
   - Small brightness or contrast variation

Avoid aggressive augmentation because ROP stage features such as demarcation lines and ridges can be subtle.

## 7. Four Classification Scenarios

| Scenario | Input | Purpose |
|---|---|---|
| S1 | Raw RGB image | Baseline without image enhancement |
| S2 | Enhanced RGB image | Test whether conservative Rahim-inspired enhancement improves classification |
| S3 | Vessel segmentation mask only | Test whether vessel structure alone is useful for staging |
| S4 | Enhanced RGB image with vessel-guided soft masking | Test whether vessel-guided image processing improves classification |

All four scenarios must use the same train, validation, and test split.

## 8. Image Enhancement for Classification

Use `Rahim2024.pdf` as the reference for classification image enhancement.

Rahim2024 emphasizes that ROP RetCam images often suffer from poor illumination, reflection, and low visibility of stage-related structures. The most relevant stage-classification finding is that restoration-style preprocessing plus CLAHE improved Stage 0-3 classification.

However, Rahim2024 used a private Calgary/McROP RetCam dataset, while this project uses the Zhao2024 dataset. Zhao2024 images were collected from Shenzhen Eye Hospital, cropped to `512x512`, and selected for clearer visualization of retina, vessels, and lesions. Because of this dataset difference, Rahim-style enhancement may improve, have little effect, or even reduce classification performance if it distorts subtle stage cues.

Therefore, the enhancement step should be framed as an experimental scenario:

```text
Does Rahim-style enhancement, originally proposed for RetCam ROP images, improve ROP stage classification on the Zhao2024 dataset?
```

Recommended implementation for this project:

1. Apply illumination or reflection correction.
2. Apply CLAHE after correction.
3. Preserve RGB output for CNN classification.

Reference method from Rahim2024:

```text
Raw RGB image
  -> DPFRr-style reflection/illumination correction
  -> CLAHE
  -> Enhanced RGB image
```

Practical implementation for this project:

```text
Raw RGB image
  -> conservative illumination/background normalization
  -> CLAHE on LAB luminance or RGB channels
  -> optional denoising
  -> Enhanced RGB image
```

The implementation should avoid aggressive restoration because Zhao2024 images may already be curated and relatively clear. The report should describe this as a Rahim-inspired enhancement, not an exact DPFRr reproduction, unless the full DPFRr method is implemented faithfully.

The analysis should compare visual examples and classification metrics to decide whether the enhancement transfers well to Zhao2024.

## 9. Vessel Segmentation Pipeline

Use `Almeida2024.pdf` as the main reference for vessel segmentation enhancement.

Recommended vessel segmentation pipeline:

```text
Raw RGB image
  -> Field-of-view mask generation
  -> CIELAB conversion
  -> CLAHE on L* channel
  -> Convert back to RGB
  -> Extract green channel
  -> Background normalization
  -> Bell-Shaped Gaussian Matched Filtering
  -> Modified Top-Hat operation
  -> Frangi vesselness filter
  -> Jerman vesselness filter
  -> Weighted Frangi + Jerman combination
  -> Triangle thresholding
  -> Small object removal
  -> Optional optic disc artifact removal
  -> Vessel mask
```

The segmentation pipeline produces an intermediate soft vesselness map:

```text
Soft vesselness map: continuous values from 0 to 1
```

This soft map is not the final segmentation output. It is thresholded and cleaned to produce the final binary vessel mask:

```text
Soft vesselness map
  -> Triangle thresholding
  -> Binary vessel mask
  -> Remove small connected components
  -> Optional morphological opening or hole filling
  -> Cleaned binary vessel mask
```

The cleaned binary vessel mask is used for segmentation evaluation and as the basis for S3 and S4 classification inputs.

For S4, do not multiply the enhanced image directly by a hard binary mask because that may remove important non-vessel stage cues. Instead, convert the cleaned binary mask into a soft vessel guide:

```text
Cleaned binary vessel mask
  -> Optional dilation
  -> Gaussian blur
  -> Soft vessel guide
  -> Apply guide to enhanced RGB image
  -> Vessel-guided enhanced RGB image
```

Example guide formula:

```text
guide = 0.5 + 0.5 * gaussian_blur(dilate(cleaned_binary_mask))
S4_image = enhanced_rgb * guide
```

This keeps S4 as a 3-channel RGB image while emphasizing vessel regions and preserving broader retinal context.

## 10. Vessel Segmentation Evaluation

Evaluate vessel segmentation using HVDROPDB vessel masks from `Agrawal2021.pdf`.

Recommended metrics:

- Dice coefficient
- Intersection over Union
- Sensitivity
- Specificity
- Accuracy

This evaluation is important because scenario S3 and S4 depend on vessel mask quality. Poor vessel masks may reduce classification performance.

The ground truth masks from Agrawal2021/HVDROPDB should be treated as binary masks. If a local ground truth file contains grayscale values because of anti-aliasing or export artifacts, binarize it before evaluation.

Recommended evaluation flow:

```text
Predicted soft vesselness map
  -> Triangle thresholding
  -> Cleaned binary prediction

Ground truth vessel mask
  -> Binarization if needed
  -> Binary ground truth

Cleaned binary prediction vs binary ground truth
  -> Dice, IoU, sensitivity, specificity, accuracy
```

The project does not create a soft ground truth. The soft vesselness map is only an intermediate prediction used to obtain a binary mask.

## 11. Classification Model

### 11.1 Main Model: Custom Tiny ResNet

Use a custom CNN as the main model so the project does not rely only on a standard pretrained architecture.

Suggested Tiny ResNet architecture:

```text
Input
  -> Conv 3x3, 32 filters
  -> Residual block x2, 32 filters
  -> Residual block x2, 64 filters, downsample
  -> Residual block x2, 128 filters, downsample
  -> Residual block x2, 256 filters, downsample
  -> Global average pooling
  -> Dropout
  -> Dense softmax, 4 classes
```

Input channels by scenario:

| Scenario | Channels |
|---|---:|
| S1 Raw RGB | 3 |
| S2 Enhanced RGB | 3 |
| S3 Vessel mask only | 1 |
| S4 Vessel-guided enhanced RGB | 3 |

For S4, the vessel mask is used in the image-processing stage to create a vessel-guided enhanced RGB image:

```text
Enhanced RGB image
  -> apply soft vessel guide derived from cleaned binary mask
  -> vessel-guided enhanced RGB image
  -> Tiny ResNet
```

### 11.2 Optional Baseline: ResNet50

Use ResNet50 as an optional comparison model because `Zhao2024.pdf` reported ResNet50 as the best baseline among tested models.

Use ResNet50 only for the RGB-based scenarios:

| Scenario | Tiny ResNet | ResNet50 |
|---|---|---|
| S1 Raw RGB | Yes | Yes |
| S2 Enhanced RGB | Yes | Yes |
| S3 Vessel mask only | Yes | No |
| S4 Vessel-guided enhanced RGB | Yes | No |

This keeps the ResNet50 baseline directly comparable with Zhao2024 because both S1 and S2 are standard 3-channel RGB inputs. It also avoids adding extra baseline complexity for the vessel-mask scenarios. Even though S4 is also a 3-channel RGB image, ResNet50 is intentionally limited to S1 and S2 so the optional baseline remains focused on raw/enhanced RGB comparison.

The custom Tiny ResNet remains the main model across all four scenarios. ResNet50 is used only to answer whether the custom model is reasonably competitive with a stronger standard CNN on RGB inputs.

## 12. Training Setup

Use the same training configuration for every scenario.

Recommended setup:

- Loss: weighted cross entropy or focal loss
- Optimizer: AdamW
- Learning rate: start around `1e-3`
- Batch size: 16 or 32
- Epochs: 50 to 100
- Early stopping: patience 10
- Model selection: best validation macro F1

Class imbalance should be handled because Stage 1 has the fewest images.

Recommended imbalance handling:

- Class-weighted loss
- Stratified split
- Report macro metrics, not only accuracy

## 13. Feature Extraction Explanation

For the assignment, feature extraction must be explained even when using CNNs.

In this project, feature extraction happens in two ways:

1. Explicit image-processing features:
   - Vessel structures from segmentation
   - Enhanced visibility of demarcation line and ridge
   - Vesselness maps from Frangi and Jerman filters

2. Learned CNN features:
   - Low-level texture and edge features in early convolution layers
   - Stage-related retinal structures in deeper residual blocks
   - Global class-discriminative representation before the softmax layer

For scenario S3, the classifier learns only from vessel geometry and vessel distribution.

For scenario S4, the classifier receives a vessel-guided enhanced RGB image. The vessel information is injected through image processing before classification, not through a special model configuration.

## 14. Evaluation Plan

Report metrics for each scenario:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision
- Per-class recall
- Confusion matrix
- Optional one-vs-rest ROC AUC

Use macro F1 as the main comparison metric because the dataset is imbalanced.

Recommended final comparison table:

| Scenario | Accuracy | Macro Precision | Macro Recall | Macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| S1 Raw | TBD | TBD | TBD | TBD | Baseline |
| S2 Enhanced | TBD | TBD | TBD | TBD | Conservative Rahim-inspired enhancement |
| S3 Vessel mask | TBD | TBD | TBD | TBD | Vessel-only information |
| S4 Vessel-guided enhanced | TBD | TBD | TBD | TBD | Enhanced image guided by cleaned vessel mask |

## 15. Analysis Questions

The report should answer these questions:

1. Does enhancement improve classification compared with raw images?
2. Which classes benefit most from enhancement?
3. Does vessel-only input contain enough information for ROP stage classification?
4. Does vessel-guided masking improve enhanced-image classification?
5. Which classes are most often confused?
6. Is Stage 1 difficult to detect because of subtle demarcation lines?
7. Does vessel segmentation quality limit S3 and S4 performance?
8. Is the custom Tiny ResNet competitive with ResNet50 on RGB scenarios S1 and S2?
9. Does the Rahim-inspired enhancement transfer well from the Rahim2024 dataset setting to the Zhao2024 dataset?

## 16. Expected Results

Expected performance trend:

```text
S1 Raw < S2 Enhanced <= S4 Vessel-Guided Enhanced
S3 Vessel Mask Only may be weaker than RGB-based scenarios
```

Reasoning:

- ROP stage classification depends on demarcation lines, ridges, and extraretinal proliferation.
- Vessel masks may help but may not contain all stage-relevant visual information.
- Enhanced RGB images should make subtle stage features easier to learn.
- Vessel-guided masking may emphasize vascular structure while preserving broader retinal context.

This trend is only a hypothesis. Because Rahim2024 and Zhao2024 use different datasets, S2 may perform worse than S1 if enhancement introduces artifacts, over-amplifies noise, or suppresses useful clinical details. That outcome would still be valuable because the assignment emphasizes process analysis, not only final accuracy.

## 17. Main Limitations

Expected limitations:

- Small dataset size, especially for Stage 1.
- Possible image-level split leakage because patient or eye identifiers are not available in the local Zhao2024 files.
- ROP stage labels can be subjective, especially Stage 1 vs Stage 2.
- Vessel segmentation generated from an external segmentation pipeline may contain errors.
- Zhao2024 dataset includes multiple orientations, and some images may not show all clinical structures clearly.
- Rahim2024 enhancement was designed on a different dataset, so transfer to Zhao2024 is uncertain.
- Full DPFRr reproduction from Rahim2024 may be difficult; this project may use a conservative Rahim-inspired adaptation instead.

## 18. Suggested Report Structure

1. Introduction
2. Dataset and problem definition
3. Related work
4. Proposed method
5. Image enhancement method
6. Vessel segmentation method
7. Classification model
8. Experimental scenarios
9. Results
10. Discussion
11. Limitations and future work
12. Conclusion

## 19. Suggested Implementation Milestones

### Week 1

- Audit dataset folders and class counts.
- Implement dataset loader and stratified split.
- Implement raw image baseline.

### Week 2

- Implement conservative Rahim-inspired enhancement.
- Implement Almeida-style vessel segmentation.
- Evaluate vessel segmentation on HVDROPDB.

### Week 3

- Train Tiny ResNet on all four scenarios.
- Optionally train ResNet50 baseline on S1 and S2 only.
- Save metrics, confusion matrices, and sample predictions.

### Week 4

- Analyze scenario differences.
- Prepare report, slides, and demo.
- Refine visualizations and final conclusions.
