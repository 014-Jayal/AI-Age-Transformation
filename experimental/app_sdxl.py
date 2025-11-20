import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import time
from datetime import date
from diffusers import StableDiffusionImg2ImgPipeline
import face_alignment
from io import BytesIO

# --- Configuration ---
# Set the device for PyTorch (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stable Diffusion Base Model and LoRA details
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # Change to an SDXL base model
LORA_MODEL_ID = "ModelsLab/age" # Set to the Hugging Face repo for age.pt
LORA_WEIGHT_NAME = "age.pt" # Set to the specific LoRA file name

# --- Streamlit App Setup ---
st.set_page_config(
    layout="centered",
    page_title="AI Age Progression Studio (All-in-One)",
    page_icon="✨", # A general magic icon
    initial_sidebar_state="expanded"
)

st.title("✨ AI Age Progression Studio (All-in-One) 👴")
st.markdown("""
Upload a photo and choose how you want to guide the AI's age transformation:
""")

# --- Helper Functions ---

# Helper function to align face 
def align_face_for_input(image_pil, fa_model, target_size=(512, 512), padding_factor=0.6):
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

        crop_dim = max(face_width, face_height)
        padded_crop_dim = crop_dim * (1 + padding_factor) # Add padding around the face

        x1 = int(center_x - padded_crop_dim / 2)
        y1 = int(center_y - padded_crop_dim / 2)
        x2 = int(center_x + padded_crop_dim / 2)
        y2 = int(center_y + padded_crop_dim / 2)

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
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to(DEVICE)
        pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME)
        st.success("Main Diffusion model and LoRA loaded.")

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=DEVICE)
        st.success("Face Alignment model loaded.")

    return pipe, fa

def perform_age_transformation_diffusion(input_image_pil, final_prompt, strength_value):
    """
    Performs age transformation using a diffusion model with a given prompt.
    """

    aligned_input_pil, _ = align_face_for_input(input_image_pil, fa_model, target_size=(512, 512))
    if aligned_input_pil is None:
        st.error("Could not align face for diffusion. Please ensure the image clearly shows a single face.")
        return None

    try:
        generator = torch.Generator(device=DEVICE).manual_seed(42) # Use a fixed seed for reproducibility

        aged_image_output = pipe(
            prompt=final_prompt,
            image=aligned_input_pil,
            num_inference_steps=50,
            guidance_scale=6.5,
            strength=strength_value,
            generator=generator,
            negative_prompt="cartoon, illustration, 3d render, anime, painting, blurry, distorted, low quality, bad anatomy, deformed, strange eyes, uncanny valley, watermark, text, signature, unnatural, extra fingers, disfigured, (gender opposite to prompt), cropped, close-up, disproportioned, multiple people, low resolution, bad quality"
        ).images[0]

    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        st.info("Tip: Try a different image or adjust the 'Transformation Strength' and prompt/settings.")
        return None

    return aged_image_output

# --- Main Streamlit App Logic ---
# Load models
pipe, fa_model = load_models()

# Sidebar for inputs
with st.sidebar:
    st.header("📸 Upload Your Photo")
    uploaded_file = st.file_uploader("Upload a clear photo of a face:", type=["jpg", "jpeg", "png"], help="For best results, use a well-lit photo with the face clearly visible.")
    
    st.header("Choose Transformation Method")
    transformation_mode = st.radio(
        "Select how you want to specify the age transformation:",
        ["Transform to Specific Age", "Transform with Custom Prompt", "Transform to Current Age (Dynamic)"],
        index=0 # Default to "Transform to Specific Age"
    )

    st.header("Common Settings")
    gender_of_person = st.radio(
        "Gender of person in photo:",
        ["Unspecified", "Male", "Female"],
        index=0,
        help="Helps the AI maintain gender identity, especially during significant age shifts. This applies to 'Specific Age' and 'Dynamic Age' modes."
    )

    strength_value = st.slider(
        "Transformation Strength (0.0 - 1.0):",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Higher values allow more dramatic changes but may alter identity. Lower values preserve identity but make subtle changes."
    )

    # --- Conditional Inputs based on Mode ---
    target_age_display = "N/A" # For display in caption later

    if transformation_mode == "Transform to Specific Age":
        st.header("🔢 Specific Age Settings")
        target_age_slider = st.slider("Select Target Age:", min_value=1, max_value=90, value=30, step=1, help="Choose the exact age you'd like the person in the photo to appear as.")
        
        # Prepare prompt for this mode
        target_age_display = f"{target_age_slider} years"
        age_descriptor = f"{target_age_slider} year old"
        if 1 <= target_age_slider <= 5: age_descriptor = "a baby" if target_age_slider < 2 else "a young child"
        elif 6 <= target_age_slider <= 12: age_descriptor = "a child"
        elif 13 <= target_age_slider <= 19: age_descriptor = "a teenager"
        elif 70 <= target_age_slider <= 90: age_descriptor = "an elderly person"
        elif target_age_slider > 90: age_descriptor = "a very old person"
        
        gender_word = "person"
        if gender_of_person == "Male": gender_word = "man"
        elif gender_of_person == "Female": gender_word = "woman"

        final_prompt_for_generation = f"a photo of {age_descriptor} {gender_word}, detailed, realistic, high quality, professional photography, studio lighting, clear face, natural skin texture"

    elif transformation_mode == "Transform with Custom Prompt":
        st.header("📝 Custom Prompt Settings")
        custom_prompt_input = st.text_input(
            "Enter your age transformation prompt:",
            value="a photo of a 70 year old man, detailed, realistic, high quality, natural skin texture",
            help="Describe the desired age and characteristics, e.g., 'a photo of an 80 year old woman with wrinkles and grey hair'. Gender and age from this prompt will override common settings."
        )
        final_prompt_for_generation = custom_prompt_input
        target_age_display = "Custom Prompt"

    elif transformation_mode == "Transform to Current Age (Dynamic)":
        st.header("⏳ Dynamic Age Settings")
        image_taken_date = st.date_input(
            "Date when the photo was taken:",
            value=date(2015, 1, 1),
            max_value=date.today(),
            help="Select the approximate date the original photo was captured."
        )
        age_in_photo = st.number_input(
            "Age of person in this photo (on that date):",
            min_value=1,
            max_value=90,
            value=20,
            step=1,
            help="Enter the age of the person depicted in the uploaded photo on the selected date."
        )

        # Calculate target age
        current_year = date.today().year
        years_diff = current_year - image_taken_date.year
        calculated_target_age = age_in_photo + years_diff

        # Display calculated age
        st.info(f"Calculated target age: **{calculated_target_age} years old**")
        target_age_display = f"{calculated_target_age} years (calculated)"

        # Prepare prompt for this mode
        age_descriptor = f"{calculated_target_age} year old"
        if 1 <= calculated_target_age <= 5: age_descriptor = "a baby" if calculated_target_age < 2 else "a young child"
        elif 6 <= calculated_target_age <= 12: age_descriptor = "a child"
        elif 13 <= calculated_target_age <= 19: age_descriptor = "a teenager"
        elif 70 <= calculated_target_age <= 90: age_descriptor = "an elderly person"
        elif calculated_target_age > 90: age_descriptor = "a very old person"
        
        gender_word = "person"
        if gender_of_person == "Male": gender_word = "man"
        elif gender_of_person == "Female": gender_word = "woman"

        final_prompt_for_generation = f"a photo of {age_descriptor} {gender_word}, detailed, realistic, high quality, professional photography, studio lighting, clear face, natural skin texture"

    st.markdown("---")
    generate_button = st.button("✨ Generate Transformed Image ✨", use_container_width=True)
    st.markdown("---")
    st.info("Developed with ❤️")


# Main content area
if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("Original Photo")
    st.image(input_image_pil, caption="Your uploaded image", use_container_width=True)

    # --- Warnings about model limitations in a dropdown ---
    with st.expander("❓ Important Notes & Limitations"):
        st.warning("""
        * **Age Range:** The AI model may struggle with very young (1-5 years) or very old (80-90+ years) transformations.
        * **Gender Consistency:** Especially with large age gaps, the model might subtly or overtly change perceived gender.
        * **Identity Preservation:** Significant transformations (high 'Strength') can alter facial features, making the generated person look less like the original.
        * **Prompt Specificity (Custom Mode):** For 'Custom Prompt' mode, the quality of the output heavily depends on how well you describe the desired age and features.
        * **Output Zoom:** If the output still appears too zoomed, try using an input image where the face is not already extremely close-up.
        """)
        st.info("Experiment with the settings! For best results, use clear, well-lit photos with a single face.")

    if generate_button:
        # Basic validation for custom prompt mode
        if transformation_mode == "Transform with Custom Prompt" and not final_prompt_for_generation.strip():
            st.error("Please enter a custom prompt for age transformation!")
        elif transformation_mode == "Transform to Current Age (Dynamic)":
            if calculated_target_age <= 0:
                st.error("Calculated age is 0 or less. Please ensure the 'Date when photo was taken' and 'Age of person in this photo' are correct.")
            elif calculated_target_age > 100:
                st.warning(f"Calculated age ({calculated_target_age}) is very high. AI models may not produce realistic results for ages above ~90. Proceeding with generation, but results may be less accurate.")
            
            # Continue with generation only if no critical error for dynamic mode
            if calculated_target_age <= 0:
                st.stop() # Stop execution if age is invalid

        with st.spinner(f"Generating transformed image..."):
            start_time = time.time()
            aged_image = perform_age_transformation_diffusion(
                input_image_pil,
                final_prompt_for_generation,
                strength_value
            )
            end_time = time.time()
            st.success(f"Image generated in {end_time - start_time:.2f} seconds!")

        if aged_image:
            caption_text = f"Transformed image (Target: {target_age_display})"
            if transformation_mode == "Transform with Custom Prompt":
                caption_text = f"Transformed image (Prompt: '{final_prompt_for_generation}')"

            st.subheader(f"Here's Your Transformed Photo!")
            st.image(aged_image, caption=caption_text, use_container_width=True)
            
            # Download button
            buf_image = BytesIO()
            aged_image.save(buf_image, format="PNG")
            byte_im = buf_image.getvalue()
            st.download_button(
                label="Download Transformed Image (512x512)",
                data=byte_im,
                file_name=f"transformed_image_{target_age_display.replace(' ', '_').replace('(','').replace(')','')}.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.error("Failed to generate transformed image. Please check your inputs and try again.")
else:
    st.info("⬆️ Upload a photo and choose a transformation method from the sidebar to get started!")