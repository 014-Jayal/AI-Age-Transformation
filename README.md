# AI Age Transformation Tool

A production-grade facial age progression and regression system built using **Stable Diffusion**, **LoRA-based age editing**, precise **face alignment**, and optional **identity verification** using ArcFace.  
This repository provides a modular, extensible framework designed for research, experimentation, and applied use in age transformation workflows.

---

## 1. Overview

The AI Age Transformation Tool enables high‑quality, identity‑preserving age modification of facial images.  
It uses a combination of:

- **Stable Diffusion (SD 1.5 / SDXL)** for generative editing  
- **LoRA adapters** specialized for age progression & regression  
- **face-alignment** for precise face centering and cropping  
- **Streamlit** for a clean, interactive UI  
- **Optional ArcFace-based similarity scoring** for evaluating identity retention

The system supports multiple transformation strategies and includes optimized variants for GPU, SDXL, and 4× high‑resolution upscaling.

---

## 2. Key Features

### 2.1 Comprehensive Age Transformation Modes
- **Specific Age Mode**  
  Generates a facial reconstruction corresponding to any target age.
- **Dynamic Age Calculation Mode**  
  Auto‑computes current age using image date and user-provided age.
- **Custom Prompt Mode**  
  Allows full prompt control for artistic or experimental age manipulation.

### 2.2 Identity Preservation
A dedicated batch-processing module uses **ArcFace embeddings** to compute similarity between original and transformed images, enabling objective assessment of identity retention.

### 2.3 Accurate Face Alignment Pipeline
Every app variant utilizes landmark detection to:
- Identify facial boundaries
- Crop and center the face region
- Normalize orientation
- Produce optimal SD inputs (512×512 or 1024×1024)

This significantly improves consistency, avoids deformities, and minimizes model drift.

### 2.4 Multiple Model Variants Included

| App Variant | Description |
|------------|-------------|
| **app.py** | Main all-in-one interface with all transformation modes |
| **app_gpu.py** | Adds Stable Diffusion **4× Upscaler** for high-resolution outputs |
| **app_sdxl.py** | Uses **SDXL** model with SDXL-compatible age LoRA |
| **app_simple.py** | A simplified lightweight demo version |
| **batch_process.py** | Batch inference with identity similarity scoring |

These modular versions make experimentation and deployment flexible.

---

## 3. Project Architecture

```
AI-Age-Transformation/
├── app.py                 # Main all-in-one Streamlit interface
├── app_gpu.py             # Upscaler-enabled GPU version
├── app_sdxl.py            # SDXL-based age transformation
├── app_simple.py          # Minimal streamlined UI
├── batch_process.py       # Batch transformations + ArcFace similarity scoring
├── requirements.txt       # Full dependency list
├── .gitignore             # Excluded large files & models
├── README.md              # Documentation
├── images/                # Sample inputs & generated outputs
└── output/                # Saved batch results
```

---

## 4. Installation

### 4.1 Clone the Repository
```bash
git clone https://github.com/014-Jayal/AI-Age-Transformation.git
cd AI-Age-Transformation
```

### 4.2 Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU acceleration, install CUDA-enabled PyTorch:
https://pytorch.org/get-started/locally/

---

## 5. Usage

### 5.1 Launch Main Application
```bash
streamlit run app.py
```

### 5.2 Launch GPU + 4× Upscaler Version
```bash
streamlit run app_gpu.py
```

### 5.3 Launch SDXL Version
```bash
streamlit run app_sdxl.py
```

### 5.4 Launch Simple Version
```bash
streamlit run app_simple.py
```

### 5.5 Batch Processing + Identity Similarity
```bash
python batch_process.py
```

Outputs will be stored under `output/`.

---

## 6. Core System Workflow

### 6.1 Face Alignment
1. Detect 2D face landmarks  
2. Compute bounding box with smart padding  
3. Crop & resize face to diffusion-friendly resolution  
4. Normalize lighting and geometry (when needed)

### 6.2 Diffusion-Based Age Editing
Stable Diffusion **Img2Img** mode is used with:
- LoRA-based age editing adapters  
- Prompt engineering  
- Guidance scaling  
- Adjustable transformation strength  

This allows controlled modification while maintaining identity.

### 6.3 Optional High-Resolution 4× Upscaling
GPU version uses:
```
stabilityai/stable-diffusion-x4-upscaler
```
to enhance outputs while maintaining detail.

### 6.4 Identity Similarity Evaluation
Using **ArcFace embeddings**:
- Extract embedding vectors from original and aged images  
- Compute cosine similarity  
- Higher scores indicate better identity preservation  

---

## 7. Example Output

| Original Image | Aged Output (60 Years) |
|----------------|------------------------|
| <img src="images/original.jpg" width="260"> | <img src="images/aged_60.jpg" width="260"> |

---

## 8. Important Notes

- Extremely young (<5) and extremely old (>80) ages may produce unpredictable results  
- Gender consistency may vary depending on prompt structure  
- Higher transformation strength creates more dramatic aging but may reduce identity fidelity  
- Lighting, camera angle, and image quality influence model performance  

---

## 9. Author

**Jayal Shah**  
GitHub: https://github.com/014-Jayal  
LinkedIn: https://www.linkedin.com/in/jayalshah04/

---

## 10. License
This project is intended for **research and educational use only**.

