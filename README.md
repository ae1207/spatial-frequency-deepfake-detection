# spatial-frequency-deepfake-detection
Final Year Project implementing a spatial-frequency deepfake detection framework using Xception-based spatial features and FFT-based spectral features, with evaluation under full-frame, face-cropped, cross-dataset, and cross-compression settings.

# Spatial-Frequency Deepfake Detection

This repository contains the implementation of a Final Year Project titled **Enhancing Deepfake Algorithms**, where enhancement refers to improving **deepfake detection algorithms** rather than deepfake generation. The project investigates a spatial-frequency feature fusion approach for deepfake detection using an Xception-based spatial branch and an FFT-based frequency branch.

The main objective is to evaluate whether combining spatial-domain features and frequency-domain features can improve deepfake detection performance compared with a spatial-only baseline model.

## Project Overview

Deepfake detection models often perform well on the dataset they are trained on, but their performance may drop when tested on unseen datasets, compressed videos, or different manipulation techniques. This project explores whether spatial-frequency feature fusion can improve detection robustness by combining:

* **Spatial features** extracted from RGB images using an Xception-based CNN.
* **Frequency features** extracted using Fast Fourier Transform (FFT).
* **Feature-level fusion** through concatenation of spatial and frequency feature vectors.
* **Binary classification** to classify samples as real or fake.

The project also compares full-frame and face-cropped input settings to determine whether face-focused preprocessing improves detection performance.

## Main Findings

The experiments show that **face-focused preprocessing has the strongest impact on detection performance**. The face-cropped models perform significantly better than the full-frame models.

The FFT-based fusion model achieved the highest AUC in the face-cropped setting, but it did not clearly outperform the Xception face-cropped baseline in accuracy, precision, or F1-score. This suggests that the current FFT-based fusion strategy provides limited additional benefit beyond strong spatial features extracted from face-cropped inputs.

## Proposed Workflow

The overall detection pipeline follows these stages:

1. Input video or image frame
2. Frame extraction
3. Face detection and cropping
4. Image resizing and normalization
5. Spatial feature extraction using Xception
6. Frequency feature extraction using FFT
7. Feature-level fusion
8. Fully connected classification layer
9. Real or fake prediction

## Model Configurations

This project evaluates four main model configurations:

| Model                 | Input Type | Description                                     |
| --------------------- | ---------- | ----------------------------------------------- |
| Xception Full-Frame   | Full frame | Spatial-only baseline using full image frames   |
| Fusion Full-Frame     | Full frame | Xception + FFT fusion using full image frames   |
| Xception Face-Cropped | Face crop  | Spatial-only baseline using cropped face images |
| Fusion Face-Cropped   | Face crop  | Xception + FFT fusion using cropped face images |

## Results Summary

| Model                 | Accuracy | Precision | Recall | F1-Score |    AUC |
| --------------------- | -------: | --------: | -----: | -------: | -----: |
| Xception Full-Frame   |   0.5982 |    0.5465 | 0.6940 |   0.6115 | 0.5356 |
| Fusion Full-Frame     |   0.6080 |    0.5571 | 0.6805 |   0.6127 | 0.5378 |
| Xception Face-Cropped |   0.8183 |    0.8019 | 0.7997 |   0.8008 | 0.8940 |
| Fusion Face-Cropped   |   0.8158 |    0.7968 | 0.8011 |   0.7989 | 0.9007 |

## Datasets

The project uses publicly available deepfake datasets for training, validation, testing, and generalization evaluation.

Datasets used include:

* FaceForensics++
* Celeb-DF
* DFDC-FOTS
* DFDC100K
* FaceForensics++ C23
* FaceForensics++ C40

FaceForensics++ is used as the main training and in-domain evaluation dataset. Other datasets are used to evaluate cross-dataset generalization and compression robustness.

## Technologies Used

* Python
* PyTorch
* torchvision
* timm
* OpenCV
* NumPy
* scikit-learn
* facenet-pytorch
* Matplotlib
* NVIDIA GPU acceleration

## Repository Structure

```text
spatial-frequency-deepfake-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── face_crops/
│
├── models/
│   ├── xception_baseline.py
│   ├── fft_branch.py
│   └── fusion_model.py
│
├── scripts/
│   ├── extract_frames.py
│   ├── crop_faces.py
│   ├── train_baseline.py
│   ├── train_fusion.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrices/
│   ├── graphs/
│   └── metrics/
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/spatial-frequency-deepfake-detection.git
cd spatial-frequency-deepfake-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Frames

```bash
python scripts/extract_frames.py --input data/raw --output data/processed
```

### 2. Perform Face Cropping

```bash
python scripts/crop_faces.py --input data/processed --output data/face_crops
```

### 3. Train Xception Baseline

```bash
python scripts/train_baseline.py --data data/face_crops --epochs 20 --batch_size 32
```

### 4. Train Spatial-Frequency Fusion Model

```bash
python scripts/train_fusion.py --data data/face_crops --epochs 20 --batch_size 32
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py --model checkpoints/best_model.pth --data data/test
```

## Evaluation Metrics

The models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Area Under the ROC Curve (AUC)
* Confusion matrix

These metrics are used to compare spatial-only detection, spatial-frequency fusion, full-frame input, face-cropped input, cross-dataset performance, and cross-compression robustness.

## Limitations

This project is limited to frame-level image-based deepfake detection. It does not include real-time deployment, audio-based detection, adversarial robustness testing, or temporal sequence modeling. The FFT-based fusion strategy used in this project is also relatively simple and may not fully capture complex spatial-frequency relationships.

## Future Work

Possible future improvements include:

* Using stronger frequency representations such as DCT, DWT, or multi-scale FFT.
* Applying attention-based or adaptive feature fusion.
* Incorporating temporal modeling for video-level detection.
* Testing robustness against adversarial attacks.
* Optimizing the model for real-time or lightweight deployment.
* Exploring audio-visual deepfake detection.

## Acknowledgement

This project was completed as part of a Final Year Project at the Faculty of Information Science and Technology, Multimedia University. Special appreciation is given to Dr. Ho Yean Li for her guidance and support throughout the project.

## Author

**Muhammad Eusoff Aminurrashid Bin Shukri**
Faculty of Information Science and Technology
Multimedia University
Melaka, Malaysia

## License

This repository is intended for academic and research purposes. Please refer to the selected license file for usage terms.
