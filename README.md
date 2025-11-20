# AI Age Transformation Tool

## Overview
The AI Age Transformation Tool is an advanced system for facial age progression and regression. It uses Stable Diffusion (v1.5) with LoRA adapters to generate realistic age transformations while preserving identity. Developed during an internship at BISAG-N (Govt. of India).

## Features
- **Multiple Transformation Modes**
  - Specific target age (e.g., 60 years old)
  - Dynamic age calculation using photo date
  - Customizable text prompts
- **Identity Preservation**
  - Face alignment preprocessing
  - InsightFace (ArcFace) similarity verification
- **Interactive Streamlit UI**

## Results Example
| Original | Age 60 Output |
|---------|----------------|
| <img src="images/original.jpg" width="280"> | <img src="images/aged_60.png" width="280"> |

## Tech Stack
- **AI:** Stable Diffusion v1.5, LoRA Age Slider, diffusers
- **CV:** face-alignment, insightface
- **Frameworks:** PyTorch, Streamlit
- **Libraries:** numpy, PIL

## Installation
```
git clone https://github.com/014-Jayal/AI-Age-Transformation.git
cd AI-Age-Transformation
pip install -r requirements.txt
```

## Usage
### Run the Web Interface
```
streamlit run app.py
```

### Batch Processing
```
python batch_process.py
```

## Project Structure
```
AI-Age-Transformation/
├── app.py
├── batch_process.py
├── requirements.txt
├── .gitignore
├── README.md
├── experimental/
└── images/
    ├── original.jpg
    └── aged_60.png
```

## Author
**Jayal Shah**

GitHub: https://github.com/014-Jayal  
LinkedIn: https://www.linkedin.com/in/jayalshah04/

## Disclaimer
This tool is intended for research and educational purposes only.
