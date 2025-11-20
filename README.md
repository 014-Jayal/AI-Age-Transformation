# AI Age Transformation Tool 📅👴

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-orange)

An advanced AI-powered tool for facial age progression and regression. This project utilizes **Stable Diffusion** with **LoRA** adapters to generate realistic age transformations while preserving the subject's identity.

Developed during an internship at **BISAG-N (Govt. of India)**.

## 🌟 Features

* **Multiple Transformation Modes:**
    * **Specific Age:** Target a precise age (e.g., "60 years old").
    * **Dynamic Age:** Upload a photo, provide the date taken, and the AI calculates the current age automatically.
    * **Custom Prompt:** Guide the aging process with detailed text descriptions.
* **Identity Preservation:** Integrates **Face Alignment** for landmark detection and **InsightFace (ArcFace)** to calculate identity similarity scores between the original and aged photos.
* **Interactive UI:** Built with **Streamlit** for easy user interaction.
* **GPU Acceleration:** Optimized for CUDA-enabled devices with optional xformers support.

## 🛠️ Tech Stack

* **Core AI:** `diffusers` (Stable Diffusion v1.5), `navmesh/Lora` (Age Slider).
* **Computer Vision:** `face-alignment` (Pre-processing), `insightface` (Identity Verification).
* **Interface:** `streamlit`.
* **Backend:** `torch`, `numpy`, `PIL`.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/014-Jayal/AI-Age-Transformation.git](https://github.com/014-Jayal/AI-Age-Transformation.git)
    cd AI-Age-Transformation
    ```

2.  **Install dependencies:**
    *Note: For `insightface`, ensure you have C++ build tools installed (Visual Studio Build Tools on Windows).*
    ```bash
    pip install -r requirements.txt
    ```
    *If you have a GPU, ensure you install the CUDA version of PyTorch.*

## 💻 Usage

### 1. Run the Web Interface
To launch the interactive dashboard:
```bash
streamlit run app.py