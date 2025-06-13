import argparse
import torch
import json
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from ruamel.yaml import YAML
from models.model_vqa import MUMC_VQA
import os
from collections import defaultdict

def load_model(args, config, tokenizer, device):
    model = MUMC_VQA(
        config=config,
        text_encoder=args.text_encoder,
        text_decoder=args.text_decoder,
        tokenizer=tokenizer
    )
    model = model.to_empty(device=device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print("✅ Loaded checkpoint:", msg)
    
    model.eval()
    return model


def preprocess_image(image_path, config):
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image {image_path}: {e}")
        return None
        
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


@torch.no_grad()
def infer_one_sample(model, image_tensor, question, answer_list, config, device):
    image_tensor = image_tensor.to(device)
    answer_list_eos = [ans + config['eos'] for ans in answer_list]
    
    topk_ids, topk_probs = model(image_tensor, [question], answer_list_eos, train=False, k=config['k_test'])
    _, pred = topk_probs[0].max(dim=0)
    return topk_ids[0][pred].item()  # Return index of predicted answer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MUMC VQA inference on a directory of images.")
    # --- Model and Config Arguments ---
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.pth).")
    parser.add_argument('--config_path', type=str, default='./configs/VQA.yaml', help="Path to the model config file (VQA.yaml).")
    parser.add_argument('--text_encoder', default='bert-base-uncased', help="Name of the text encoder model.")
    parser.add_argument('--text_decoder', default='bert-base-uncased', help="Name of the text decoder model.")
    parser.add_argument('--device', default='cuda', help="Device to run inference on (e.g., 'cuda' or 'cpu').")
    
    # --- Input/Output Path Arguments ---
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file to save results.")
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load config
    yaml = YAML(typ='safe')
    with open(args.config_path, 'r') as f:
        config = yaml.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # Load model
    model = load_model(args, config, tokenizer, device)

    # --- Use input_dir from args ---
    image_groups = defaultdict(list)
    
    print(f"🔍 Scanning for images in: {args.input_dir}")
    for filename in os.listdir(args.input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == "IMG" and parts[1].startswith("ENC"):
            encounter_id = parts[1]
            image_path = os.path.join(args.input_dir, filename)
            image_groups[encounter_id].append(image_path)

    if not image_groups:
        print("❌ No images found in the specified format (e.g., IMG_ENCxxxxx_...jpg). Please check the --input_dir path and filenames.")
        return

    # Sort images in each group
    for encounter_id in image_groups:
        image_groups[encounter_id].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # (Your questions_and_answers dictionary remains here)
    questions_and_answers = {
        "CQID010-001": ("How much of the body is affected?", ["single spot", "limited area", "widespread", "Not mentioned"]),
        # ... (all other questions are here)
        "CQID036-001": ("What is the skin lesion texture?", ["smooth", "rough", "Not mentioned"]),
    }

    results = []
    total_encounters = len(image_groups)
    for i, encounter_id in enumerate(sorted(image_groups.keys())):
        encounter_image_paths = image_groups[encounter_id]
        print(f"🔄 Processing encounter {i+1}/{total_encounters}: {encounter_id} with {len(encounter_image_paths)} images...")
        
        question_predictions = {}
        
        for image_path in encounter_image_paths:
            img_tensor = preprocess_image(image_path, config)
            if img_tensor is None:
                continue # Skip corrupted images

            for question_id, (question, answer_list) in questions_and_answers.items():
                answer_index = infer_one_sample(model, img_tensor, question, answer_list, config, device)
                
                if question_id not in question_predictions:
                    question_predictions[question_id] = []
                question_predictions[question_id].append(answer_index)
        
        encounter_result = {"encounter_id": encounter_id}
        for question_id, indices in question_predictions.items():
            if not indices:
                # Handle case where no valid images were processed for this encounter
                final_answer = questions_and_answers[question_id][1].index("Not mentioned") # Default to "Not mentioned"
            else:
                counts = defaultdict(int)
                for idx in indices:
                    counts[idx] += 1
                # Find the index with the highest count (majority vote)
                final_answer = max(counts, key=counts.get)
            
            encounter_result[question_id] = final_answer
        
        results.append(encounter_result)

    # --- Save result to output_file from args ---
    # 1. Get the directory part of the output file path
    output_dir = os.path.dirname(args.output_file)
    # 2. If the directory is not empty, create it
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # 3. Save the file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved successfully to: {args.output_file}")


if __name__ == '__main__':
    main()
