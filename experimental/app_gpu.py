import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import time
# Import both pipelines
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline
import face_alignment
from io import BytesIO

# --- Configuration ---
# Set the device for PyTorch (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stable Diffusion Base Model and LoRA details for initial generation
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_ID = "navmesh/Lora"
LORA_WEIGHT_NAME = "age_slider-LECO-v1.safetensors"

# Upscaler Model ID
UPSCALE_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

# --- Streamlit App Setup ---
st.set_page_config(layout="centered", page_title="AI Age Progression")
st.title("AI Age Progression 🚀")
st.write("Upload a photo, choose a target age, and see how AI imagines you might look!")

# --- Helper Functions ---

# Helper function to align face
def align_face_for_input(image_pil, fa_model, target_size=(512, 512), padding_factor=0.3):
    try:
        preds = fa_model.get_landmarks(np.array(image_pil))

        if preds is None or len(preds) == 0:
            st.warning("Warning: No face detected by face_alignment model during initial alignment.")
            return None, None

        landmarks = preds[0]

        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        face_width = x_max - x_min
        face_height = y_max - y_min

        crop_size = max(face_width, face_height)
        padded_crop_size = crop_size * (1 + padding_factor)

        x1 = int(center_x - padded_crop_size / 2)
        y1 = int(center_y - padded_crop_size / 2)
        x2 = int(center_x + padded_crop_size / 2)
        y2 = int(center_y + padded_crop_size / 2)

        img_width, img_height = image_pil.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        cropped_image = image_pil.crop((x1, y1, x2, y2))
        aligned_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
        return aligned_image, (x1, y1, x2, y2)

    except Exception as e:
        st.error(f"Error during face alignment: {e}")
        return None, None

# Core Age Progression Logic
@st.cache_resource
def load_models():
    """Loads diffusion pipeline, face alignment model, and Upscale pipeline, caches them."""
    with st.spinner("Loading AI models... This might take a moment (especially the first time)."):
        # Load Diffusion Model (for initial generation)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to(DEVICE)
        pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME)
        st.success("Main Diffusion model and LoRA loaded.")

        # Load Face Detection and Alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=DEVICE)
        st.success("Face Alignment model loaded.")

        # Load Stable Diffusion Upscale Pipeline
        # This will download the x4 upscaler model.
        with st.spinner("Loading Diffusion-based Upscaler model..."):
            upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(
                UPSCALE_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            upscale_pipe = upscale_pipe.to(DEVICE)
            st.success("Diffusion-based Upscaler model loaded.")

    return pipe, fa, upscale_pipe # Return all loaded models

def perform_age_transformation_diffusion(input_image_pil, target_age_number, pipe, fa_model, custom_prompt_suffix="person"):
    """
    Performs age transformation using a diffusion model.
    """
    target_prompt = f"a photo of a {target_age_number} year old {custom_prompt_suffix}, detailed, realistic, high quality"

    st.info(f"Generating image for target age: {target_age_number}")

    aligned_input_pil, _ = align_face_for_input(input_image_pil, fa_model, target_size=(512, 512))
    if aligned_input_pil is None:
        st.error("Could not align face for diffusion. Please ensure the image clearly shows a face.")
        return None

    try:
        generator = torch.Generator(device=DEVICE).manual_seed(42)

        aged_image_output = pipe(
            prompt=target_prompt,
            image=aligned_input_pil,
            num_inference_steps=50,
            guidance_scale=6.5,
            strength=0.6,
            generator=generator,
            negative_prompt="cartoon, illustration, 3d render, anime, painting, blurry, distorted, low quality, bad anatomy, deformed, strange eyes, uncanny valley, watermark, text"
        ).images[0]

    except Exception as e:
        st.error(f"Error during diffusion inference: {e}")
        return None

    return aged_image_output

# --- Main Streamlit App Logic ---
pipe, fa, upscale_pipe = load_models() # Get all models from cached function

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(input_image_pil, caption="Original Image", use_container_width=True)

    st.sidebar.header("Age Progression Options")
    target_age_choice = st.sidebar.slider("Select Target Age", min_value=1, max_value=90, value=30, step=1)

    upscale_output = st.sidebar.checkbox("Upscale Output (4x)", value=True) # Now 4x upscale
    
    prompt_suffix = "person" # Keeping custom_prompt_suffix as 'person'

    if st.sidebar.button("Generate Aged Image"):
        if input_image_pil:
            with st.spinner("Aging your photo... this will take a moment."):
                aged_image = perform_age_transformation_diffusion(
                    input_image_pil,
                    target_age_choice,
                    pipe,
                    fa,
                    custom_prompt_suffix=prompt_suffix
                )

            if aged_image:
                st.subheader("Generated Aged Image (512x512)")
                st.image(aged_image, caption=f"Aged to {target_age_choice} years", use_container_width=True)
                
                # Download button for the 512x512 image
                buf_original_res = BytesIO()
                aged_image.save(buf_original_res, format="PNG")
                byte_im_original_res = buf_original_res.getvalue()
                st.download_button(
                    label="Download Original Resolution Image",
                    data=byte_im_original_res,
                    file_name=f"aged_to_{target_age_choice}_512x512.png",
                    mime="image/png"
                )

                if upscale_output:
                    with st.spinner("Upscaling image (4x)... this will take a few seconds and requires GPU."):
                        # Convert PIL image to a suitable tensor format for VAE encoding
                        # VAE expects [0,1] range, then it normalizes to [-1,1] internally
                        # The upscale pipeline's low_res_latents should come from the VAE of your *original* pipe.
                        
                        # 1. Resize the image to 128x128 as the upscaler expects a low-res input
                        # The stabilityai/stable-diffusion-x4-upscaler internally resizes to 128x128
                        # We pass the 512x512 image as 'image' for conditioning, but also latents.
                        
                        # For StableDiffusionUpscalePipeline, it uses an internal VAE for low_res_latents.
                        # You just need to provide the low-resolution PIL image directly as 'image'
                        # It will automatically downsample and then use that for the upscaling process.
                        
                        # Let's resize the 512x512 image to 128x128 for the low-res input to the upscaler
                        low_res_img = aged_image.resize((128, 128), Image.Resampling.LANCZOS)
                        
                        # Define a more detailed prompt for upscaling
                        upscale_prompt = f"a high-quality, detailed photo of a {target_age_choice} year old {prompt_suffix}, 4k, 8k, photorealistic"

                        upscaled_image_output = upscale_pipe(
                            prompt=upscale_prompt,
                            image=low_res_img, # This is the low-resolution input for the upscaler
                            num_inference_steps=75, # Can use more steps for better quality
                            guidance_scale=9.0, # Stronger guidance for upscaling
                            generator=torch.Generator(device=DEVICE).manual_seed(42)
                        ).images[0]

                    st.subheader("Upscaled Image (4x)")
                    st.image(upscaled_image_output, caption=f"Upscaled to {upscaled_image_output.width}x{upscaled_image_output.height}", use_container_width=True)

                    # Download button for upscaled image
                    buf_upscale = BytesIO()
                    upscaled_image_output.save(buf_upscale, format="PNG")
                    byte_im_upscale = buf_upscale.getvalue()
                    st.download_button(
                        label="Download Upscaled Image",
                        data=byte_im_upscale,
                        file_name=f"aged_to_{target_age_choice}_upscaled_4x.png",
                        mime="image/png"
                    )
            else:
                st.warning("Could not generate aged image. Please try a different photo or age.")
        else:
            st.warning("Please upload an image first.")
else:
    st.info("Upload an image to get started!")

# Optional: Add a simple footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by BISAG")