import json
import os
import torch
from PIL import Image
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
from tqdm import tqdm
import argparse # << THÊM THƯ VIỆN

# --- Helper Functions ---
# (Các hàm helper không thay đổi, giữ nguyên)

def encode_image_to_base64(image_path):
    """Encode an image file to base64 with data URI prefix."""
    try:
        _, ext = os.path.splitext(image_path)
        mime_type = f"image/{ext.lower().strip('.')}"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded_data = base64.b64encode(binary_data)
            base64_string = base64_encoded_data.decode('utf-8')
            return f"data:{mime_type};base64,{base64_string}", None
    except FileNotFoundError:
        return None, f"Error: Image file not found at {image_path}"
    except Exception as e:
        return None, f"Error encoding image {image_path}: {e}"

def load_image(image_path):
    """Load and preprocess image from file path."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image, None
    except FileNotFoundError:
        return None, f"Error: Image file not found at {image_path}"
    except Exception as e:
        return None, f"Error loading image {image_path}: {e}"

def init_BLIP_model(model_name, device):
    """Initialize and return BLIP model and processor."""
    print(f"Loading BLIP model: {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    print(f"BLIP model loaded on {device}")
    return model, processor

def generate_caption(model, processor, image, max_length, prompt):
    """Generate caption for an image using BLIP."""
    # Lưu ý: Prompt hiện không được sử dụng trực tiếp trong hàm generate của BLIP-base
    # Nó được giữ lại ở đây nếu bạn muốn sử dụng một mô hình hỗ trợ prompt trong tương lai
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
            length_penalty=1.0,
            repetition_penalty=1.2,
            temperature=0.9
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# --- Định nghĩa Argument Parser ---
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate captions for dermatological images using BLIP model.")

    # Các đối số bắt buộc
    parser.add_argument("--input-json", type=str, required=True,
                        help="Path to the input JSON file containing image IDs.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Base path to the directory containing image files.")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Path to save the output JSON file with generated captions.")

    # Các đối số tùy chọn với giá trị mặc định
    parser.add_argument("--model-name", type=str, default="Salesforce/blip-image-captioning-base",
                        help="Name of the BLIP model from Hugging Face Hub.")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum length of the generated caption.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to run the model on ('auto', 'cuda', 'cpu').")

    args = parser.parse_args()

    # Xử lý logic cho device='auto'
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# --- Main Logic ---
def main(args):
    """Main function to process JSON, generate captions with BLIP, and save results."""
    
    # PROMPT được giữ lại như một hằng số trong code
    PROMPT = """
    ## Role and Goal:
    You are "DermCaption AI," a highly specialized Large Language Model functioning as an expert in dermatological image analysis and description. Your primary objective is to generate exceptionally accurate, detailed, and clinically relevant captions for dermatological images, adhering strictly to observational description based only on visual evidence. You are participating in an image caption challenge where performance is judged on precision, completeness, appropriate use of terminology, and adherence to instructions. Your goal is to achieve the highest possible score by producing captions suitable for expert review (e.g., by dermatologists).
    ## Core Task:
    Analyze the provided dermatological image and generate a concise yet comprehensive descriptive caption. Focus exclusively on the visual findings present in the image.
    ## Input:
    You will receive a single dermatological image. Assume the image is intended for clinical assessment or educational purposes.
    ## Output Requirements & Constraints:
    Focus on Observation: Describe only what is visually apparent in the image. Do not infer patient history, symptoms (like itch or pain), potential causes, or provide differential diagnoses or definitive diagnoses.
    Precise Dermatological Terminology: Utilize accurate and standard dermatological terms for:
    Primary Lesions: (e.g., macule, patch, papule, plaque, nodule, tumor, vesicle, bulla, pustule, cyst, wheal).
    Secondary Lesions: (e.g., scale, crust, erosion, ulcer, fissure, atrophy, lichenification, scar, excoriation).
    Color: (e.g., erythematous, violaceous, hyperpigmented, hypopigmented, yellowish, flesh-colored, white, black). Specify variations if present.
    Morphology/Shape: (e.g., annular, arcuate, linear, round, oval, polygonal, umbilicated, targetoid).
    Arrangement/Configuration: (e.g., grouped, clustered, scattered, discrete, confluent, dermatomal, Blaschkoid, serpiginous, reticular).
    Distribution/Location: Describe the affected anatomical area(s) as precisely as possible based on the image field of view (e.g., "dorsal forearm," "left cheek," "interdigital space," "generalized," "localized to extensor surfaces").
    Texture/Surface Characteristics: (e.g., smooth, verrucous, scaly, crusted, ulcerated, smooth, shiny, dull).
    Borders: (e.g., well-demarcated, ill-defined, regular, irregular, raised).
    Structure and Detail:
    Begin with the most prominent feature(s) or lesion type(s).
    Systematically describe the relevant characteristics listed above.
    Include an estimation of size (e.g., "approximately 2 cm papule," "plaques ranging from 1-3 cm") only if scale is reasonably inferable or highly relevant to the morphology (avoid guessing if unclear).
    If multiple lesion types are present, describe each clearly.
    Clarity and Conciseness: Be precise and avoid jargon where simpler standard terms suffice, but prioritize correct medical terminology. Aim for a descriptive paragraph format.
    Objectivity: Maintain a neutral, objective, and clinical tone. Avoid subjective language (e.g., "ugly," "looks like").
    Handling Ambiguity: If image quality limits the assessment of certain features (e.g., blurriness, poor lighting), you may note this cautiously (e.g., "Surface texture is difficult to assess due to image resolution"). Do not invent details.
    Format: Present the final caption as a single, coherent block of text.
    ## Example Structure (Conceptual - Do Not Copy Verbatim):
    "[Location, e.g., On the dorsal aspect of the right forearm,] there is/are [Number/Arrangement, e.g., multiple discrete and confluent] [Primary Lesion(s), e.g., erythematous papules and plaques] measuring [Size Range, e.g., from 0.5 cm to 3 cm in diameter]. The lesions exhibit [Shape/Borders, e.g., irregular shapes with well-demarcated borders] and possess [Surface Characteristics/Secondary Lesions, e.g., overlying silvery scale]. [Additional features, e.g., Some central clearing is noted in the larger plaques. No vesicles or pustules are visualized]."
    ## Final Instruction:
    Proceed with analyzing the provided image and generate the best possible descriptive caption according to these guidelines to maximize your performance in the challenge. Your expertise in precise dermatological description is critical.
    """

    # Initialize the BLIP model and processor using arguments
    try:
        model, processor = init_BLIP_model(args.model_name, args.device)
    except Exception as e:
        print(f"Critical error initializing BLIP model: {e}")
        return

    # Load input JSON data using argument path
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {args.input_json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input_json}")
        return
    except Exception as e:
        print(f"Unidentified error when reading JSON file: {e}")
        return

    # Initialize empty dictionary for final output
    final_output_data = {}
    total_images_processed = 0
    total_captions_generated = 0

    # Process each encounter in the JSON data
    for encounter_index, encounter in enumerate(input_data):
        encounter_id = encounter.get("encounter_id", f"UNKNOWN_ENCOUNTER_{encounter_index}")
        image_ids = encounter.get("image_ids", [])
        print(f"\nProcessing Encounter: {encounter_id} ({encounter_index + 1}/{len(input_data)})")

        # Process each image ID in the encounter
        for image_index, image_id in enumerate(image_ids):
            # Construct image path using argument
            image_path = os.path.join(args.image_dir, image_id)
            print(f"  Processing Image: {image_id} ({image_index + 1}/{len(image_ids)}) - Total: {total_images_processed + 1}")

            # Check if image exists and load it
            if not os.path.exists(image_path):
                error_msg = f"Error: Image file does not exist at {image_path}"
                print(f"    {error_msg}")
                total_images_processed += 1
                continue

            image, error = load_image(image_path)
            if error:
                print(f"    {error}")
                total_images_processed += 1
                continue

            try:
                print(f"    Generating caption with BLIP...")
                start_time = time.time()

                # Generate caption with BLIP using arguments
                caption = generate_caption(model, processor, image, args.max_length, PROMPT)

                # Post-process caption if needed
                prefix_patterns = [
                    "This is a", "This image shows", "The image shows",
                    "In this image", "The dermatological image shows"
                ]
                for pattern in prefix_patterns:
                    if caption.lower().startswith(pattern.lower()):
                        caption = caption[len(pattern):].strip()
                        if caption and caption[0].islower():
                            caption = caption[0].upper() + caption[1:]

                end_time = time.time()
                print(f"    Caption generated in {end_time - start_time:.2f} seconds")
                print(f"    Caption: {caption}")

                # Create key and value in the desired format and add to main dictionary
                output_value = {
                    "image": image_id,
                    "caption": caption
                }
                final_output_data[image_id] = output_value

                total_captions_generated += 1

            except Exception as e:
                error_message = f"Error generating caption for {image_id}: {e}"
                print(f"    {error_message}")

            total_images_processed += 1

    # Save final_output_data to JSON file using argument path
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, indent=4, ensure_ascii=False)
        print(f"\nProcessed a total of {total_images_processed} images.")
        print(f"Successfully generated {total_captions_generated} captions.")
        print(f"Results successfully saved to {args.output_json}")
    except Exception as e:
        print(f"\nError saving results to {args.output_json}: {e}")

# Run the main function if script is executed directly
if __name__ == "__main__":
    # Phân tích các đối số từ dòng lệnh
    arguments = parse_arguments()
    # Chạy logic chính với các đối số đã được phân tích
    main(arguments) 