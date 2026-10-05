# BAS Filtering for Generative Augmentation in Pneumonia Detection

Task-aligned filtering of DCGAN-generated chest X-rays to improve YOLOv8 pneumonia detection on the RSNA dataset.

## Overview

Only 22.5% of patients in the RSNA Pneumonia Detection Challenge data are pneumonia-positive. This project generates extra synthetic positive CXRs with a **DCGAN** and keeps only the useful ones using a proposed **Bbox Alignment Score (BAS)**:

```
BAS = IoU(B_pred, B_cond)
```

`B_cond` is a bounding box sampled from the RSNA bbox distribution, and `B_pred` is the box predicted by a frozen YOLOv8-m detector on the generated image. Only images with `BAS >= 0.35` are added to the training set. Unlike global realism metrics such as FID, BAS checks whether a synthetic image gives the detector spatially correct supervision.

## Splits

All splits are trained with YOLOv8-m and evaluated on the same real validation set (5,337 patients).

| Split | Training data |
|-------|---------------|
| A | Real only (baseline) |
| B | Real + traditional augmentation |
| C | Real + 10,000 unfiltered DCGAN images |
| D | Real + 787 BAS-filtered DCGAN images |

## Results

| Split | mAP@0.5 | mAP@0.5:0.95 | Recall | Precision | F1 |
|-------|:-------:|:------------:|:------:|:---------:|:--:|
| A: Real only | 0.3551 | 0.1460 | 0.4168 | 0.3699 | 0.3920 |
| B: Traditional aug. | 0.3411 | 0.1319 | **0.4316** | 0.3795 | 0.4039 |
| C: DCGAN (unfiltered) | 0.3449 | 0.1410 | 0.4051 | 0.3760 | 0.3900 |
| **D: DCGAN + BAS** | **0.3705** | **0.1498** | 0.4125 | **0.4014** | **0.4069** |

- Split D gives a **4.3% relative mAP@0.5 gain** over the baseline.
- Unfiltered DCGAN data (Split C) does not help, so the gain comes from BAS filtering.
- Only 787 of 10,000 generated images (7.9%) passed the threshold.

## Setup

```bash
pip install pydicom opencv-python-headless ultralytics scikit-learn
```

Download the [RSNA Pneumonia Detection Challenge](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) data separately. It is not included in this repository.

## Limitations

- The DCGAN is not spatially conditioned, so BAS filters after generation rather than guiding it.
- Evaluation uses a single validation split, and the threshold (0.35) was not ablated.

Future work: threshold ablation and a ControlNet-conditioned diffusion model.

## Authors

Aieza Noor, Muhammad Nouman Noor
Department of AI & Data Science, FAST NUCES Islamabad, Pakistan
