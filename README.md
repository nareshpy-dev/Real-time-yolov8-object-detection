# Real-Time Aerial Object Detection using YOLOv8 (DOTA Dataset)

## Overview

This project implements a high-resolution multi-class aerial object detection system using YOLOv8m trained on the DOTA dataset. The system detects and localizes multiple object categories in large-scale aerial imagery with stable convergence and strong precision.

The project covers the complete pipeline:

* Dataset preprocessing
* Oriented Bounding Box (OBB) to YOLO format conversion
* Model training and optimization
* Performance evaluation
* Inference pipeline implementation

---

## Dataset Details

* Dataset: DOTA (Dataset for Object Detection in Aerial Images)
* Number of Classes: 15
* Training Images: 1,411
* Validation Images: 458
* Total Annotated Objects: 69,551
* Image Resolution: 1024 × 1024

---

## Model Configuration

* Model: YOLOv8m
* Framework: Ultralytics YOLOv8 (PyTorch backend)
* Optimizer: AdamW
* Epochs: 100
* Mixed Precision Training (AMP): Enabled
* GPU Used: NVIDIA A100 (80GB)

---

## Final Performance Metrics

* Precision: 0.797
* Recall: 0.531
* mAP@0.5: 0.586
* mAP@0.5:0.95: 0.393

The model achieved stable convergence across 100 epochs on high-resolution aerial imagery and demonstrated strong precision in multi-class detection.

---

## Key Contributions

* Designed and implemented an end-to-end object detection pipeline.
* Converted oriented bounding box annotations into YOLO format.
* Identified and corrected non-normalized and corrupted label issues.
* Optimized large-resolution (1024px) training using GPU acceleration and AMP.
* Built inference workflow for multi-object detection on aerial images.
* Evaluated performance using mAP metrics, precision-recall analysis, and validation curves.

---

## Tech Stack
* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Matplotlib

---

## How to Run

### Install Dependencies

```
pip install ultralytics opencv-python torch numpy matplotlib
```

### Train the Model

```
yolo detect train data=data.yaml model=yolov8m.pt imgsz=1024 epochs=100
```

### Run Inference

```
yolo detect predict model=best.pt source=your_image.jpg
```

---

## Repository Structure

```
├── YOLOv8monDOTAv1_5.ipynb
├── README.md
├── requirements.txt
└── results/
```

---
-->Sample Results:
<img width="628" height="299" alt="testimage2" src="https://github.com/user-attachments/assets/aa5d6e5c-5856-4070-8b64-541d11acd3e9" />
<img width="686" height="361" alt="model predictions" src="https://github.com/user-attachments/assets/943859e7-9f9e-43f1-88e6-e0f5a0fbba7a" />
<img width="365" height="226" alt="Ground truth Vs predicted" src="https://github.com/user-attachments/assets/f8d7f3e7-ee84-42e9-bce7-b64ec1f43b18" />
<img width="359" height="224" alt="Ground truth Vs predicted2" src="https://github.com/user-attachments/assets/0b021097-5b78-4fa6-97d7-2b308c9223fd" />
<img width="412" height="338" alt="testimage1" src="https://github.com/user-attachments/assets/98bac452-28a2-4149-a6b5-7422e08c3052" />

## Future Improvements

* Hyperparameter tuning to improve recall
* Model compression for edge deployment
* ONNX or TensorRT export for faster inference
* Deployment via Streamlit for real-time web demo

---

If you want, I can now give you a shorter, resume-optimized README version that looks even sharper for product-based companies.
