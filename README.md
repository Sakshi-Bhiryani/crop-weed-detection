#  Automated Crop & Weed Detection Pipeline

An end-to-end computer vision system engineered for precision agriculture. By moving away from traditional blanket chemical application, this project leverages a high-speed **YOLOv8** deep learning architecture to accurately classify and locate invasive weeds versus commercial crops in real-time, providing immediate spatial telemetry for automated target spraying.



##  Architectural Pillars

The design architecture decomposes machine learning operations, data curation, and edge deployment into four distinct structural zones:

# Pillar 1: Automated Storage Sanitization (`1_organize.py`)
* Establishes a systematic local directory map dividing assets into isolated `train`, `val`, and `test` data structures.
* Executes precise file-system partitioning logic, automatically sorting raw image assets and bounding-box coordinates into optimal evaluation ratios.
* Maps directory pathways dynamically through a centralized dataset blueprint configuration (`data.yaml`).

### Pillar 2: Spatial Feature Expansion (`utils.py`)
* Deploys custom computer vision processing pipelines to augment data quality.
* Utilizes mathematical transformation matrices to artificially expand the training dataset from 546 cleaned frames to 1,300 augmented images.
* Automatically recalculates bounding-box coordinates during spatial flips and adjustments, preventing training plateau and improving model resilience.

###  Pillar 3: Neural Optimization & Core Training (`train_model.py`)
* Compiles and executes training cycles on high-performance deep-learning neural networks.
* Fine-tunes the **YOLOv8** object tracking architecture to balance high inference speeds with spatial coordinate accuracy.
* Serializes final, fully optimized mathematical coefficients into production-ready network weights (`best.pt`).

###  Pillar 4: Edge Inference Telemetry Dashboard (`4_app.py`)
* Delivers a lightweight, client-facing graphical terminal built specifically for real-time field diagnostics.
* Caches heavy model resources directly into active RAM buffers to achieve low-latency target evaluation during image or video stream uploads.
* Translates bounding array outputs into a visual framework, calculating localized weed indexes for agricultural equipment operators.



##  Software & Dependency Stack

The backend framework and data pipeline are engineered completely with **Python** and standard open-source machine learning infrastructure:

* **Object Detection Core:** `Ultralytics YOLOv8` & `PyTorch` (Neural network design, loss optimization, and prediction tracking)
* **Client Interface Delivery:** `Streamlit` (Web-based application deployment, real-time widget rendering)
* **Array Mathematics & Processing:** `NumPy` & `Pillow` (Multi-dimensional coordinate array logic, image pixel transformations)
* **Computer Vision Tools:** `OpenCV-Python` (Spatial image expansions, color-space adjustments, and telemetry drawing)



## 📁 Project Repository Geometry Map

```text
crop_weed_project/
├── dataset/                  # Split dataset directory
│   ├── train/                # Training matrix arrays (Images & YOLO Labels)
│   ├── val/                  # Validation check arrays (Images & YOLO Labels)
│   └── test/                 # Isolated test evaluation frames (Images & YOLO Labels)
├── metrics/                  # Statistical performance and logging nodes
│   ├── results.png           # Loss and precision optimization curves
│   └── confusion_matrix.png  # Class prediction error tracking layout
├── 1_organize.py             # Data engineering partitioner & dataset split script
├── utils.py                  # Data augmentation toolkit & coordinate re-calculation system
├── train_model.py            # Neural network compilation and YOLOv8 training script
├── 4_app.py                  # Live interactive Streamlit UI dashboard execution script
├── best.pt                   # Serialized production-ready neural weights file
├── data.yaml                 # Core data path configuration mapping schema
└── requirements.txt          # Absolute environment software dependency manifest