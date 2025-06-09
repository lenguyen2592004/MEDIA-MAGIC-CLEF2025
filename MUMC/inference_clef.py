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
    image = Image.open(image_path).convert("RGB")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='./configs/VQA.yaml')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load config
    yaml = YAML(typ='safe')
    with open(args.config_path, 'r') as f:
        config = yaml.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # Load model
    model = load_model(args, config, tokenizer, device)

    # Group images by encounter_id
    path = '/kaggle/input/imageclefmed-mediqa-magic-2025/images_final/images_final/images_valid'
    image_groups = defaultdict(list)
    
    for filename in os.listdir(path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Parse encounter_id from filename (format: IMG_ENC00908_00001.jpg)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == "IMG" and parts[1].startswith("ENC"):
            encounter_id = parts[1]
            image_path = os.path.join(path, filename)
            image_groups[encounter_id].append(image_path)

    # Sort images in each group by image number
    for encounter_id in image_groups:
        image_groups[encounter_id].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # image_paths = sorted(image_paths)
    
    questions_and_answers = {
    "CQID010-001": ("How much of the body is affected?", ["single spot", "limited area", "widespread", "Not mentioned"]),
    "CQID011-001": ("1 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID011-002": ("2 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID011-003": ("3 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID011-004": ("4 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID011-005": ("5 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID011-006": ("6 Where is the affected area?", ["head", "neck", "upper extremities", "lower extremities", "chest/abdomen", "back", "other (please specify)", "Not mentioned"]),
    "CQID012-001": ("1 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID012-002": ("2 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID012-003": ("3 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID012-004": ("4 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID012-005": ("5 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID012-006": ("6 How large are the affected areas? Please specify which affected area for each selection.", ["size of thumb nail", "size of palm", "larger area", "Not mentioned"]),
    "CQID015-001": ("When did the patient first notice the issue?", ["within hours", "within days", "within weeks", "within months", "over a year", "multiple years", "Not mentioned"]),
    "CQID020-001": ("1 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-002": ("2 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-003": ("3 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-004": ("4 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-005": ("5 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-006": ("6 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-007": ("7 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-008": ("8 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID020-009": ("9 What label best describes the affected area?", ["raised or bumpy", "flat", "skin loss or sunken", "thick or raised", "thin or close to the surface", "warty", "crust", "scab", "weeping", "Not mentioned"]),
    "CQID025-001": ("Is there any associated itching with the skin problem?", ["yes", "no", "Not mentioned"]),
    "CQID034-001": ("Compared to the normal surrounding skin, what is the color of the skin lesion?", ["normal skin color", "pink", "red", "brown", "blue", "purple", "black", "white", "combination (please specify)", "hyperpigmentation", "hypopigmentation", "Not mentioned"]),
    "CQID035-001": ("How many skin lesions are there?", ["single", "multiple (please specify)", "Not mentioned"]),
    "CQID036-001": ("What is the skin lesion texture?", ["smooth", "rough", "Not mentioned"]),
}

    results = []
    for encounter_id in sorted(image_groups.keys()):
        encounter_image_paths = image_groups[encounter_id]
        print(f"Processing encounter {encounter_id} with {len(encounter_image_paths)} images...")
        
        # Dictionary to collect predictions for each question across images
        question_predictions = {}
        
        for image_path in encounter_image_paths:
            for question_id, (question, answer_list) in questions_and_answers.items():
                img_tensor = preprocess_image(image_path, config)
                answer_index = infer_one_sample(model, img_tensor, question, answer_list, config, device)
                
                if question_id not in question_predictions:
                    question_predictions[question_id] = []
                question_predictions[question_id].append(answer_index)
        
        # Determine the final answer for each question using majority vote
        encounter_result = {"encounter_id": encounter_id}
        for question_id, indices in question_predictions.items():
            # Count occurrences of each index
            counts = {}
            for idx in indices:
                counts[idx] = counts.get(idx, 0) + 1
            # Find the index with the highest count
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            final_index = candidates[0]  # Handle ties by selecting the first
            encounter_result[question_id] = final_index
        
        results.append(encounter_result)

    # Save result
    os.makedirs("/kaggle/working/output/clef2025", exist_ok=True)
    with open("/kaggle/working/output/clef2025/mumc_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved mumc_test.json")


if __name__ == '__main__':
    main()
