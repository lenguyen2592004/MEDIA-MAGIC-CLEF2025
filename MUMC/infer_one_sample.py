# infer_one_sample.py

import argparse
import torch
import json
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.model_vqa import MUMC_VQA
from ruamel.yaml import YAML
import os
import ast  # for parsing list from string

def load_model(args, config, tokenizer, device):
    model = MUMC_VQA(config=config,
                     text_encoder=args.text_encoder,
                     text_decoder=args.text_decoder,
                     tokenizer=tokenizer)
    model = model.to_empty(device=device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    msg = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint:", msg)
    model.eval()
    return model

def preprocess_image(image_path, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image).unsqueeze(0)

@torch.no_grad()
def infer_one_sample(model, image_tensor, question, answer_list, config, device):
    image_tensor = image_tensor.to(device)
    answer_list_eos = [ans + config['eos'] for ans in answer_list]
    topk_ids, topk_probs = model(image_tensor, [question], answer_list_eos, train=False, k=config['k_test'])
    _, pred = topk_probs[0].max(dim=0)
    return answer_list[topk_ids[0][pred]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='./configs/VQA.yaml')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    
    # New: Accept answer list via file or CLI
    parser.add_argument('--answers_path', type=str, help='Path to JSON file containing answer list')
    parser.add_argument('--answers', type=str, help='List of answers as string, e.g., \'["cat", "dog", "rabbit"]\'')
    
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config_path, 'r') as f:
        config = yaml.load(f)

    # Load answer list
    if args.answers_path:
        with open(args.answers_path, 'r') as f:
            config['answer_list'] = json.load(f)
    elif args.answers:
        config['answer_list'] = ast.literal_eval(args.answers)
    else:
        raise ValueError("You must provide either --answers_path or --answers")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    # Load model
    model = load_model(args, config, tokenizer, device)

    # Process image and infer
    image_tensor = preprocess_image(args.image_path, config)
    answer = infer_one_sample(model, image_tensor, args.question, config['answer_list'], config, device)

    print(f"Q: {args.question}")
    print(f"A: {answer}")

if __name__ == '__main__':
    main()
