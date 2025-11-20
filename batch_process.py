import os
import torch
import numpy as np
from PIL import Image
import json
import time
from diffusers import StableDiffusionImg2ImgPipeline
import face_alignment
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import torchvision.transforms as transforms # Used for potential model specific transforms

# --- Configuration ---
# Set the device for PyTorch (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path to your input image (relative to main.py)
INPUT_IMAGE_PATH = 'images/test.jpg'
# Directory to save output images
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

# Stable Diffusion Base Model and LoRA details
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_ID = "navmesh/Lora"
LORA_WEIGHT_NAME = "age_slider-LECO-v1.safetensors"

# --- Helper Functions ---

# Helper function to align face 
def align_face_for_input(image_pil, fa_model, target_size=(512, 512), padding_factor=0.3):
    try:
        preds = fa_model.get_landmarks(np.array(image_pil))

        if preds is None or len(preds) == 0:
            print("Warning: No face detected by face_alignment model.")
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
        print(f"Error during face alignment: {e}")
        return None, None

# Helper function to get ArcFace embedding 
def get_arcface_embedding(app, image_pil):
    try:
        img_np = np.array(image_pil)[:, :, ::-1] # RGB to BGR
        faces = app.get(img_np)

        if len(faces) == 0:
            print("Warning: No face detected by ArcFace for embedding.")
            return None
        return faces[0].embedding
    except Exception as e:
        print(f"Error getting ArcFace embedding: {e}")
        return None

# Helper function for cosine similarity 
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# --- Core Age Progression Logic ---
def perform_age_transformation_diffusion(input_image_pil, target_age_prompt, pipe, fa_model, identity_app=None):
    print(f"\n--- Attempting Age Transformation with Diffusion (Target: {target_age_prompt}) ---")
    start_time = time.time()

    # Align input image for the model
    # Note: Changed target_size to (640,640) for potentially better ArcFace detection
    # but Stable Diffusion v1.5 prefers 512x512. If results are poor, change back to (512,512)
    # and focus on getting clear input images.
    aligned_input_pil, _ = align_face_for_input(input_image_pil, fa_model, target_size=(512, 512))
    if aligned_input_pil is None:
        print("Could not align face for diffusion. Skipping.")
        return None, 0.0

    original_embedding = None
    if identity_app:
        original_embedding = get_arcface_embedding(identity_app, aligned_input_pil)
        if original_embedding is None:
            print("Warning: Could not get original face embedding.")

    print(f"Performing diffusion inference for age transformation: {target_age_prompt}")
    try:
        generator = torch.Generator(device=DEVICE).manual_seed(42)

        aged_image_output = pipe(
            prompt=target_age_prompt,
            image=aligned_input_pil,
            num_inference_steps=50,
            guidance_scale=5.0,
            strength=0.6, # Experiment with this value (0.0 to 1.0)
            generator=generator,
            negative_prompt="cartoon, illustration, 3d render, anime, painting, blurry, distorted, low quality, bad anatomy, deformed, strange eyes, uncanny valley, watermark, text"
        ).images[0]

        print(f"Diffusion inference took: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error during diffusion inference: {e}")
        return None, 0.0

    aged_image_pil = aged_image_output

    aged_embedding = None
    similarity = 0.0
    if identity_app:
        aged_embedding = get_arcface_embedding(identity_app, aged_image_pil)
        if original_embedding is not None and aged_embedding is not None:
            similarity = cosine_similarity(original_embedding, aged_embedding)
            print(f"Identity Similarity (Original vs. Aged): {similarity:.4f}")
        else:
            print("Could not compute identity similarity due to missing embeddings.")

    return aged_image_pil, similarity


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Age Progression Project ---")

    # --- 1. Load Pre-trained Diffusion Model & Auxiliaries ---
    print("Loading Diffusion Model...")
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 on GPU, float32 on CPU
        )
        pipe = pipe.to(DEVICE)
        print(f"Base model '{BASE_MODEL_ID}' loaded.")

        pipe.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME)
        print("LoRA weights loaded and applied.")

    except Exception as e:
        print(f"Error loading Diffusion Pipeline or LoRA: {e}")
        print("Please ensure you have accepted the model's license on Hugging Face (if required) for the base model.")
        exit(1) # Exit if critical models can't load

    print("\nLoading Face Detection and Alignment model (face-alignment)...")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=DEVICE)
    print("Face Alignment model loaded.")

    print("\nLoading ArcFace model (for identity verification)...")
    try:
        app = FaceAnalysis(name='buffalo_l')
        # ctx_id=0 for GPU, -1 for CPU. Use -1 if DEVICE is 'cpu'
        app.prepare(ctx_id=0 if DEVICE == 'cuda' else -1, det_size=(640, 640))
        print("ArcFace model loaded for identity verification.")
    except Exception as e:
        print(f"Error loading ArcFace model: {e}")
        print("Please ensure `insightface` and `onnxruntime` are installed correctly.")
        app = None # Set to None so the identity check is skipped later


    print("All models initialized successfully.")

    # --- 2. Load Input Image ---
    input_image_pil = None
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at '{INPUT_IMAGE_PATH}'. Please place your image there.")
    else:
        try:
            input_image_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
            print(f"Input image '{INPUT_IMAGE_PATH}' loaded successfully.")
            # Optionally display original image (requires matplotlib)
            plt.imshow(input_image_pil)
            plt.title("Original Image")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error loading input image: {e}")

    if input_image_pil is not None:
        # --- 3. Define Target Age Prompts ---
        target_age_prompts = [
            "a high-quality photograph of a young man, clear skin, vibrant eyes, natural lighting, studio portrait",
            "a highly detailed photograph of a middle-aged man, subtle wrinkles, calm expression, professional studio lighting",
            "a photorealistic image of an elderly person, deep lines, white hair, gentle smile, warm lighting, natural skin texture"
            
        ]

        print("\nStarting age progression for the loaded image with various prompts...")

        results = []
        for i, prompt in enumerate(target_age_prompts):
            print(f"\nProcessing for prompt: '{prompt}'")
            aged_image, similarity = perform_age_transformation_diffusion(
                input_image_pil,
                prompt,
                pipe,
                fa,
                app
            )
            if aged_image:
                results.append({
                    "prompt": prompt,
                    "image": aged_image,
                    "similarity": similarity
                })
                # Save the generated image
                output_filename = os.path.join(OUTPUT_DIR, f"aged_image_{i+1}_{prompt.replace(' ', '_').replace('/', '_')}.png")
                aged_image.save(output_filename)
                print(f"Generated image saved to: {output_filename}")
            else:
                print(f"Could not generate image for prompt: '{prompt}'")

        # --- 4. Display Results ---
        print("\n--- Age Progression Results ---")
        if not results:
            print("No aged images were successfully generated.")
        else:
            print("Original Image:")
            plt.imshow(input_image_pil)
            plt.title("Original Image")
            plt.axis('off')
            plt.show()

            for res in results:
                print(f"\nPrompt: '{res['prompt']}'")
                print(f"Identity Similarity: {res['similarity']:.4f}")
                plt.imshow(res['image'])
                plt.title(f"Aged Image: {res['prompt']} (Similarity: {res['similarity']:.2f})")
                plt.axis('off')
                plt.show()

        print("\nAge progression process complete. Check the 'output' folder for saved images!")
    else:
        print("Skipping age progression due to missing or unreadable input image.")