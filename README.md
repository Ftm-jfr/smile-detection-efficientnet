# Smile Detection with EfficientNet-B3

Binary smile / no-smile classification on face images, plus a real-time inference script that runs on a video file or a phone camera stream.

This project was developed as a course project for **Deep Learning** at the **University of Isfahan**.

## Overview

- **Task:** classify a face image as *Smile* or *No Smile*.
- **Model:** EfficientNet-B3 (ImageNet-pretrained, via `timm`) with a new 2-class head, partially fine-tuned.
- **Data:** GENKI-4k (about 4,000 face images labeled smile / non-smile).
- **Preprocessing:** face detection and alignment with dlib, crops resized to 300x300.
- **Training:** raw plus augmented images, best checkpoint chosen by validation accuracy.

## Results

| Split | Accuracy |
|---|---|
| Best validation | 92.33% |
| Test (600 images) | **91.33%** (548 / 600) |

Test-set classification report: non-smile precision 0.91 / recall 0.90, smile precision 0.91 / recall 0.93 (macro F1 0.91). The notebook contains the full report and confusion matrix.

## Repository contents

| File | Description |
|---|---|
| `train.ipynb` | Training and evaluation notebook (data preparation, alignment, augmentation, fine-tuning, metrics). |
| `test.py` | Quick demo script for trying the saved weights on a video: Haar-cascade face detection, then smile classification (label 1 = smile), drawn on the video frames. |

## Dataset

GENKI-4k is a public smile-detection dataset. The version used here was obtained from the course / Kaggle. Please download it separately and adjust the dataset paths at the top of the notebook (the notebook was written for a Kaggle environment, so paths look like `/kaggle/input/...`).

## Model weights

The trained weights are included in this repository as `best_model.pth`. The file is the checkpoint saved by the notebook: a dictionary with the keys `model_state`, `optimizer_state`, `epoch` and `val_acc`, for a `timm` `efficientnet_b3` with 2 output classes. Keep it next to `test.py`.

## Setup

```bash
pip install torch torchvision timm "opencv-python<5" pillow numpy scikit-learn matplotlib dlib face_recognition albumentations jupyter
```

`dlib` needs a C++ compiler and CMake to build on most systems. The notebook downloads dlib's 5-point landmark model (`shape_predictor_5_face_landmarks.dat`) automatically. Only the notebook needs `dlib`, `face_recognition` and `albumentations`; `test.py` needs `torch`, `torchvision`, `timm`, `opencv-python` and `pillow`.

Use the regular `opencv-python` package, pinned below version 5: `opencv-python-headless` has no window support, so `cv2.imshow` fails, and OpenCV 5.x does not provide `cv2.CascadeClassifier`, which `test.py` uses for face detection. Do not install both packages in the same environment.

## Running

**Training / evaluation:** open `train.ipynb` (Kaggle or locally with a GPU), fix the dataset paths, and run the cells in order.

**Real-time inference:** edit `test.py`:

- Choose the video source. The default is a video file; replace `path/to/your_video.mp4` with your own file. To use a phone camera over Wi-Fi (for example with the IP Webcam app) or a laptop webcam, comment out the first `cv2.VideoCapture(...)` line and uncomment option 2 (`http://<phone-ip>:8080/video`) or option 3 (`0`). Only one source should be active.
- Weights: the script loads `best_model.pth` from the same folder.

Then run `python test.py`. Press `q` to quit.

## Notes and limitations

- The reported accuracy comes from a single evaluation on the notebook's held-out test split (600 images).
- `test.py` is a quick demo: it finds faces with a Haar cascade and resizes the crop to 300x300, without the notebook's dlib alignment, so live predictions can be less accurate than the offline evaluation. The demo's accuracy has not been measured.
- GENKI-4k contains mostly frontal, posed faces, so performance on unconstrained video (strong head rotation, low light, partial occlusion) will be lower.
- Haar-cascade detection is fast but misses non-frontal faces.
- Results come from a single train/validation/test split, with no cross-validation.

## References

- Tan and Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019. https://arxiv.org/abs/1905.11946
- GENKI-4k dataset, MPLab, UCSD.
