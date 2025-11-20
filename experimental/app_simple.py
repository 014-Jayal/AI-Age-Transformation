import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import time
from diffusers import StableDiffusionImg2ImgPipeline
import face_alignment
from io import BytesIO

# --- Configuration ---
# Set the device for PyTorch (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stable Diffusion Base Model and LoRA details for initial generation
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_ID = "navmesh/Lora"
LORA_WEIGHT_NAME = "age_slider-LECO-v1.safetensors"

# --- Streamlit App Setup ---
st.set_page_config(
    layout="centered", # Can be "wide" if you prefer more horizontal space
    page_title="AI Age Progression Studio",
    page_icon="👶", # A fun emoji icon
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

st.title("👶 AI Age Progression Studio 👴")
st.markdown("""
Welcome! Upload a photo and let our AI imagine how you might look at a different age.
This app uses a Stable Diffusion model with a LoRA for age progression.
""")

# --- Helper Functions ---

# Helper function to align face
def align_face_for_input(image_pil, fa_model, target_size=(512, 512), padding_factor=0.3):
    """Aligns a face in the input image for better diffusion results."""
    try:
        preds = fa_model.get_landmarks(np.array(image_pil))

        if preds is None or len(preds) == 0:
            st.warning("Warning: No face detected by face_alignment model. Please ensure the image clearly shows a face.")
            return None, None

        landmarks = preds[0] # Assuming one face is detected

        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        face_width = x_max - x_min
        face_height = y_max - y_min

        crop_size = max(face_width, face_height)
        padded_crop_size = crop_size * (1 + padding_factor) # Add padding around the face

        x1 = int(center_x - padded_crop_size / 2)
        y1 = int(center_y - padded_crop_size / 2)
        x2 = int(center_x + padded_crop_size / 2)
        y2 = int(center_y + padded_crop_size / 2)

        # Ensure crop coordinates are within image bounds
        img_width, img_height = image_pil.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        cropped_image = image_pil.crop((x1, y1, x2, y2))
        aligned_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        return aligned_image, (x1, y1, x2, y2)

    except Exception as e:
        st.error(f"An error occurred during face alignment: {e}")
        return None, None

# Core Model Loading Logic
@st.cache_resource
def load_models():
    """Loads diffusion pipeline and face alignment model, caches them."""
    with st.spinner("Loading AI models... This might take a moment (especially the first time)."):
        # Load Diffusion Model (for initial generation)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to(DEVICE)
        pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME)
        # xformers is commented out as per previous issues. If you install it, uncomment below.
        # if DEVICE == 'cuda':
        #    pipe.enable_xformers_memory_efficient_attention()
        st.success("Main Diffusion model and LoRA loaded.")

        # Load Face Detection and Alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=DEVICE)
        st.success("Face Alignment model loaded.")

    return pipe, fa # Only return pipe and fa

def perform_age_transformation_diffusion(input_image_pil, target_age_number, pipe, fa_model, gender_of_person, strength_value, custom_prompt_suffix="person"):
    """
    Performs age transformation using a diffusion model.
    """
    # Determine the gender suffix for the prompt
    gender_word = ""
    if gender_of_person == "Male":
        gender_word = "man"
    elif gender_of_person == "Female":
        gender_word = "woman"
    else:
        gender_word = "person" # Default if unspecified

    # Optional: Add age-specific descriptors for better results at extremes
    age_descriptor = f"{target_age_number} year old"
    if 1 <= target_age_number <= 5:
        age_descriptor = "a baby" if target_age_number < 2 else "a young child"
    elif 6 <= target_age_number <= 12:
        age_descriptor = "a child"
    elif 13 <= target_age_number <= 19:
        age_descriptor = "a teenager"
    elif 70 <= target_age_number <= 90:
        age_descriptor = "an elderly person"
    elif target_age_number > 90:
        age_descriptor = "a very old person"

    target_prompt = f"a photo of {age_descriptor} {gender_word}, detailed, realistic, high quality, professional photography"
    
    st.info(f"Generating image for target age: {target_age_number} with strength {strength_value}...")

    aligned_input_pil, _ = align_face_for_input(input_image_pil, fa_model, target_size=(512, 512))
    if aligned_input_pil is None:
        st.error("Could not align face for diffusion. Please ensure the image clearly shows a single face.")
        return None

    try:
        generator = torch.Generator(device=DEVICE).manual_seed(42) # Use a fixed seed for reproducibility

        aged_image_output = pipe(
            prompt=target_prompt,
            image=aligned_input_pil,
            num_inference_steps=50,
            guidance_scale=6.5,
            strength=strength_value, # Use the dynamic strength value
            generator=generator,
            negative_prompt="cartoon, illustration, 3d render, anime, painting, blurry, distorted, low quality, bad anatomy, deformed, strange eyes, uncanny valley, watermark, text, signature, unnatural, extra fingers, disfigured"
        ).images[0]

    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        st.info("Tip: Try a different image or adjust the 'Strength' and 'Target Age' settings.")
        return None

    return aged_image_output

# --- Main Streamlit App Logic ---
# Load models (only pipe and fa)
pipe, fa = load_models()

# Sidebar for inputs
with st.sidebar:
    st.header("📸 Upload Your Photo")
    uploaded_file = st.file_uploader("Upload a clear photo of a face:", type=["jpg", "jpeg", "png"], help="For best results, use a well-lit photo with the face clearly visible.")
    
    st.header("🔢 Age Progression Settings")
    
    target_age_choice = st.slider("Select Target Age:", min_value=1, max_value=90, value=30, step=1, help="Choose the age you'd like the person in the photo to appear as.")
    
    # New: Gender selection
    gender_of_person = st.radio(
        "Gender of person in photo:",
        ["Unspecified", "Male", "Female"],
        index=0, # Default to Unspecified
        help="Helps the AI maintain gender identity, especially at extreme ages."
    )

    # New: Strength slider
    strength_value = st.slider(
        "Transformation Strength (0.0 - 1.0):",
        min_value=0.0,
        max_value=1.0,
        value=0.6, # Default value
        step=0.05,
        help="Higher values allow more dramatic changes but may alter identity. Lower values preserve identity but make subtle changes."
    )

    st.markdown("---")
    generate_button = st.button("✨ Generate Aged Image ✨", use_container_width=True)
    st.markdown("---")
    st.info("Developed with ❤️")


# Main content area
if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Original Photo")
    st.image(input_image_pil, caption="Your uploaded image", use_container_width=True)

    # --- Warnings about model limitations ---
    st.warning("""
    **Important Note on AI Limitations:**
    * **Extreme Ages (1-20 & 80-90):** The AI model may struggle to produce realistic or consistent results at very young or very old ages due to limitations in its training data for these ranges.
    * **Gender Handling:** In some cases, especially at extreme ages, the model might subtly or overtly change the perceived gender of the person. Using the "Gender of person in photo" setting can help mitigate this.
    * **Identity Preservation:** While efforts are made to preserve identity, significant age transformations (especially with higher 'Strength') can sometimes alter facial features, making the generated person look less like the original.
    """)
    st.info("Experiment with the 'Transformation Strength' and 'Gender of person in photo' settings for better results!")

    if generate_button:
        with st.spinner("Processing your image and generating the aged version..."):
            start_time = time.time()
            aged_image = perform_age_transformation_diffusion(
                input_image_pil,
                target_age_choice,
                pipe,
                fa,
                gender_of_person, # Pass the gender choice
                strength_value # Pass the strength value
            )
            end_time = time.time()
            st.success(f"Image generated in {end_time - start_time:.2f} seconds!")

        if aged_image:
            st.subheader(f"Here's Your Photo Aged to {target_age_choice} Years Old!")
            st.image(aged_image, caption=f"Generated image at {target_age_choice} years", use_container_width=True)
            
            # Download button for the generated image
            buf_image = BytesIO()
            aged_image.save(buf_image, format="PNG")
            byte_im = buf_image.getvalue()
            st.download_button(
                label="Download Aged Image (512x512)",
                data=byte_im,
                file_name=f"aged_to_{target_age_choice}_years.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.error("Failed to generate aged image. Please check the input image and try again.")
else:
    st.info("⬆️ Upload a photo using the sidebar to get started!")