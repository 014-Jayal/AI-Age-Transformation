
# AI Age Transformation Tool

A comprehensive facial age progression and regression system built using **Stable Diffusion**, **LoRA age models**, **face-alignment**, and **identity verification** (ArcFace).  
This repository includes multiple variants of the app—GPU-optimized, SDXL-based, upscaler-enabled, and a simplified demo version.

---

## Key Features

### 1. Multiple Age Transformation Modes  
Supported in the main Streamlit app (all-in-one version) fileciteturn0file1:
- **Specific Age Transformation**  
- **Dynamic Age Calculation** (uses date taken + current date)  
- **Custom Prompt Transformation**

### 2. Identity Preservation  
Batch backend uses ArcFace to compute similarity between original and generated faces fileciteturn0file2.

### 3. Face Alignment  
All app variants use `face_alignment` for landmark detection and cropping, ensuring the diffusion model receives clean face inputs fileciteturn0file1.

### 4. Multiple Model Variants
Available app versions:
- **app.py** – Main all‑in‑one UI with 3 transformation modes fileciteturn0file1
- **app_gpu.py** – Adds SD‑Upscaler (4× upscaling for high‑quality results) fileciteturn0file5
- **app_sdxl.py** – Uses SDXL base + separate LoRA model for age editing fileciteturn0file6
- **app_simple.py** – Minimal streamlined age editing UI fileciteturn0file7
- **batch_process.py** – Backend script for batch inference + identity similarity scoring fileciteturn0file2

---

## Project Architecture

```
AI-Age-Transformation/
├── app.py                 # Main all-in-one Streamlit interface
├── app_gpu.py             # Adds SD Upscaler 4x for enhanced output
├── app_sdxl.py            # SDXL-based version of the app
├── app_simple.py          # Lightweight simplified UI
├── batch_process.py       # CLI backend for batch processing & ArcFace similarity
├── requirements.txt       # Project dependencies
├── .gitignore             # Excludes images, models, venv, cache etc.
├── README.md              # Documentation
├── images/
│   ├── original.jpg
│   └── aged_*.png
└── output/
    └── (batch processed results)
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/014-Jayal/AI-Age-Transformation.git
cd AI-Age-Transformation
```

### 2. Install Requirements
The project uses Stable Diffusion and multiple face-processing libraries:  
Dependencies from `requirements.txt` fileciteturn0file4:

```bash
pip install -r requirements.txt
```

If you have a GPU, install CUDA-enabled PyTorch from:  
https://pytorch.org/get-started/locally/

---

## Usage

### **1. Run the Main Streamlit App**
```bash
streamlit run app.py
```

### **2. GPU + Upscaling Version**
```bash
streamlit run app_gpu.py
```

### **3. SDXL Version**
```bash
streamlit run app_sdxl.py
```

### **4. Simple Version**
```bash
streamlit run app_simple.py
```

### **5. Batch Processing**
Executes age transformation for defined prompts & computes ArcFace similarity:

```bash
python batch_process.py
```

Outputs are saved in the `output/` directory.

---

## Core Workflow

### 1. Face Alignment
All apps use `face_alignment` to detect landmarks and crop faces cleanly before diffusion.  
Ensures stable and centered generation.  
Example function: `align_face_for_input` fileciteturn0file1

### 2. Diffusion Age Transformation  
Uses Stable Diffusion (v1.5 or SDXL) in Img2Img mode with LoRA injected:  
```python
pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME)
```
(Example from app.py) fileciteturn0file1

### 3. Optional Upscaling (4×)
Using `stabilityai/stable-diffusion-x4-upscaler` in app_gpu.py fileciteturn0file5

---

## Example Outputs

| Original | Aged Output |
|---------|-------------|
| <img src="images/original.jpg" width="280"> | <img src="images/aged_60.jpg" width="280"> |

---

## Important Notes
- Extreme ages (1–5, 80+) may produce inconsistent results depending on model training distribution.
- Gender consistency can drift during large age transitions — prompt engineering helps.
- Identity preservation varies with transformation strength.

---

## Author

**Jayal Shah**  
GitHub: https://github.com/014-Jayal  
LinkedIn: https://www.linkedin.com/in/jayalshah04/

---

## License
This tool is intended for **research and educational purposes only**.

