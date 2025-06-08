import google.generativeai as genai
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from dotenv import load_dotenv
import argparse
import random

# Tải biến môi trường từ file .env (nếu có)
load_dotenv()

def initialize_model(model_name: str, api_key: str):
    """Khởi tạo và cấu hình model Gemini."""
    final_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not final_api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            final_api_key = user_secrets.get_secret("GEMINI_API_KEY")
        except (ImportError, Exception):
            pass

    if not final_api_key:
        raise ValueError("Gemini API Key not found. Provide it via --api_key, .env file, or Kaggle Secrets.")

    genai.configure(api_key=final_api_key)

    generation_config = {
        "temperature": 0.2, "top_p": 1.0, "top_k": 32,
        "max_output_tokens": 8192, "response_mime_type": "application/json",
    }
    safety_settings = [
        {"category": c, "threshold": "BLOCK_NONE"} for c in 
        ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
         "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    ]
    
    print(f"Initializing model: {model_name}")
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

def load_data(file_path: Path):
    """Tải dữ liệu từ file JSON."""
    if not file_path or not file_path.exists():
        print(f"Warning: File not found or path is None: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def build_clinical_context(encounter_id, ht_dict):
    """Xây dựng chuỗi văn bản ngữ cảnh lâm sàng từ dữ liệu Health Topic."""
    encounter_data = ht_dict.get(encounter_id)
    if not encounter_data: return "No clinical context available."
    context = "### Clinical Context\n"
    context += f"**Patient Query:** {(encounter_data.get('query_content_en') or 'N/A').strip()}\n\n"
    if responses := encounter_data.get('responses'):
        context += "**Doctor's Discussion:**\n"
        for i, response in enumerate(responses):
            context += f"- Response {i+1}: {(response.get('content_en') or 'N/A').strip()}\n"
    return context

def build_questions_text(questions):
    """Xây dựng phần văn bản câu hỏi và lựa chọn."""
    prompt_part = ""
    for q_data in questions:
        prompt_part += f"**Question ID:** {q_data['qid']}\n"
        prompt_part += f"**Question:** {q_data['question']}\n"
        prompt_part += "**Options:**\n"
        for i, ans in enumerate(q_data['answer']):
            prompt_part += f"{i}: {ans}\n"
        prompt_part += "---\n"
    return prompt_part

def build_vqa_prompt(clinical_context, questions_for_encounter):
    """Xây dựng prompt ZERO-SHOT hoàn chỉnh cho một ca bệnh."""
    prompt = f"""
You are an expert dermatologist analyzing a clinical case. Your task is to answer a series of multiple-choice questions based on the provided clinical context and images.
You must choose only one option for each question. Your final output must be a single JSON object. The keys of this object should be the question IDs (e.g., "CQID010-001"), and the values must be the integer index of your chosen answer.
**Do not add any text, explanations, or markdown formatting like ```json before or after the JSON object.**

Here is the clinical context:
{clinical_context}
---
**Questions to Answer:**
{build_questions_text(questions_for_encounter)}
"""
    return prompt

def build_fewshot_prompt(examples, query_context, query_questions):
    """Xây dựng prompt FEW-SHOT bằng cách sử dụng các ví dụ đã có."""
    prompt = """
You are an expert dermatologist. You will be shown a few example cases with their correct answers, followed by a new case that you must solve.
For each case, analyze the clinical context and images to answer the multiple-choice questions.
Your final output for the new case must be a single JSON object. The keys should be the question IDs and the values must be the integer index of the chosen answer.
**Do not add any text or markdown formatting before or after the final JSON object.**

---
"""
    # Xây dựng phần ví dụ
    for i, ex in enumerate(examples):
        prompt += f"--- EXAMPLE {i+1} START ---\n"
        prompt += ex['context']
        prompt += "\n**Questions and Options:**\n"
        prompt += build_questions_text(ex['questions'])
        prompt += "**Correct Answer (JSON Format):**\n"
        prompt += json.dumps(ex['ground_truth'], indent=2) + "\n"
        prompt += f"--- EXAMPLE {i+1} END ---\n\n"
    
    # Xây dựng phần nhiệm vụ mới
    prompt += "--- YOUR TASK START ---\n"
    prompt += "Now, analyze the following new case and provide your answers in the same JSON format.\n\n"
    prompt += query_context
    prompt += "\n**Questions to Answer:**\n"
    prompt += build_questions_text(query_questions)
    prompt += "**Your Answer (JSON only):**\n"
    
    return prompt

def load_images(image_ids, image_dir: Path):
    """Tải hình ảnh từ danh sách ID và trả về list các đối tượng PIL Image."""
    images = []
    for img_id in image_ids:
        img_path = image_dir / img_id
        if img_path.exists():
            try:
                images.append(Image.open(img_path))
            except Exception as e:
                print(f"Warning: Could not open image {img_path}. Error: {e}")
        else:
            print(f"Warning: Image file not found: {img_path}")
    return images

def prepare_fewshot_examples(vqa_path, ht_path, gt_path):
    """Tải và chuẩn bị dữ liệu few-shot."""
    if not all([vqa_path, ht_path, gt_path]):
        print("One or more few-shot data files not provided. Running in zero-shot mode.")
        return []

    print("\nLoading few-shot example data...")
    vqa_data_fs = load_data(vqa_path)
    ht_data_fs = load_data(ht_path)
    gt_data_fs = load_data(gt_path)

    if not all([vqa_data_fs, ht_data_fs, gt_data_fs]):
        print("Could not load all necessary few-shot files. Running in zero-shot mode.")
        return []

    ht_dict_fs = {item['encounter_id']: item for item in ht_data_fs}
    gt_dict_fs = {item['encounter_id']: item for item in gt_data_fs}

    encounters_fs = {}
    for item in vqa_data_fs:
        enc_id = item['encounter_id']
        if enc_id not in encounters_fs:
            encounters_fs[enc_id] = {'questions': [], 'image_ids': item['image_ids']}
        encounters_fs[enc_id]['questions'].append(item)
    
    prepared_examples = []
    for enc_id, data in encounters_fs.items():
        if enc_id not in gt_dict_fs:
            # print(f"Warning: No ground truth found for few-shot example {enc_id}. Skipping.")
            continue # Bỏ qua nếu không có câu trả lời đúng
        
        # Tạo bản sao ground truth và xóa encounter_id để chỉ còn lại các cặp qid:answer
        gt_answer = gt_dict_fs[enc_id].copy()
        gt_answer.pop('encounter_id', None)
        
        example = {
            'encounter_id': enc_id,
            'context': build_clinical_context(enc_id, ht_dict_fs),
            'questions': data['questions'],
            'image_ids': data['image_ids'],
            'ground_truth': gt_answer
        }
        prepared_examples.append(example)
        
    print(f"Successfully prepared {len(prepared_examples)} few-shot examples.")
    return prepared_examples


def run_vqa_task(args):
    """Hàm chính thực hiện toàn bộ quy trình VQA."""
    vqa_path = Path(args.vqa_file)
    ht_path = Path(args.ht_file)
    img_dir = Path(args.image_dir)
    out_path = Path(args.output_file)
    
    # Chuẩn bị dữ liệu few-shot
    fewshot_examples = prepare_fewshot_examples(
        Path(args.fewshot_vqa_file) if args.fewshot_vqa_file else None,
        Path(args.fewshot_ht_file) if args.fewshot_ht_file else None,
        Path(args.fewshot_gt_file) if args.fewshot_gt_file else None
    )

    print("\nInitializing model...")
    model = initialize_model(args.model_name, args.api_key)
    
    print("\nStarting Dermatology VQA Task...")
    print(f"VQA file: {vqa_path}")
    print(f"Image directory: {img_dir}")
    print(f"Output file: {out_path}")
    print(f"Mode: {'Few-shot (' + str(args.num_shots) + ' examples)' if fewshot_examples else 'Zero-shot'}")

    print("\nLoading main datasets...")
    vqa_data = load_data(vqa_path)
    ht_data = load_data(ht_path)

    if not vqa_data or not ht_data:
        print("Could not load necessary main data files. Exiting.")
        return

    ht_dict = {item['encounter_id']: item for item in ht_data}
    encounters = {}
    for item in vqa_data:
        enc_id = item['encounter_id']
        if enc_id not in encounters:
            encounters[enc_id] = {'questions': [], 'image_ids': item['image_ids']}
        encounters[enc_id]['questions'].append(item)

    final_output_list = []
    
    print(f"\nProcessing {len(encounters)} encounters...")
    for encounter_id, data in tqdm(encounters.items(), desc="Encounters"):
        try:
            query_context = build_clinical_context(encounter_id, ht_dict)
            query_images = load_images(data['image_ids'], img_dir)
            
            if not query_images:
                print(f"\nWarning: No valid images for encounter {encounter_id}. Skipping.")
                continue

            prompt_parts = []
            
            # Chọn và xây dựng prompt dựa trên chế độ few-shot hoặc zero-shot
            if fewshot_examples and args.num_shots > 0:
                # Chọn ngẫu nhiên k ví dụ (không bao gồm ca bệnh hiện tại nếu nó có trong ví dụ)
                examples_to_use = [ex for ex in fewshot_examples if ex['encounter_id'] != encounter_id]
                k = min(args.num_shots, len(examples_to_use))
                selected_examples = random.sample(examples_to_use, k)
                
                prompt_text = build_fewshot_prompt(selected_examples, query_context, data['questions'])
                prompt_parts.append(prompt_text)

                # Thêm hình ảnh từ các ví dụ
                fs_img_dir = Path(args.fewshot_image_dir)
                for ex in selected_examples:
                    prompt_parts.extend(load_images(ex['image_ids'], fs_img_dir))
            else:
                prompt_text = build_vqa_prompt(query_context, data['questions'])
                prompt_parts.append(prompt_text)

            # Luôn thêm hình ảnh của ca bệnh cần giải quyết vào cuối
            prompt_parts.extend(query_images)

            response = model.generate_content(prompt_parts, request_options={'timeout': 180})
            
            try:
                encounter_answers = json.loads(response.text)
                result_item = {'encounter_id': encounter_id}
                result_item.update(encounter_answers)
                final_output_list.append(result_item)
            except json.JSONDecodeError:
                print(f"\nError: Failed to decode JSON for encounter {encounter_id}.\nModel response: {response.text}")
            except Exception as e:
                print(f"\nError processing response for {encounter_id}: {e}")

            time.sleep(1.5) # Giữ khoảng nghỉ để tránh rate limiting

        except Exception as e:
            print(f"\nCritical error processing encounter {encounter_id}: {e}")
            continue
            
    print(f"\nProcessing complete. Saving {len(final_output_list)} results to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_list, f, indent=4)

    print("Task finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dermatology VQA task with optional few-shot learning.")
    
    # Tham số cho dữ liệu chính (cần dự đoán)
    parser.add_argument('--vqa_file', type=str, required=True, help='Path to the main VQA JSON dataset file.')
    parser.add_argument('--ht_file', type=str, required=True, help='Path to the main Health Topic JSON file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the main image directory.')
    parser.add_argument('--output_file', type=str, default='generated_results.json', help='Path to save the final JSON output.')

    # Tham số cho mô hình và API
    parser.add_argument('--model_name', type=str, default='gemini-1.5-flash-latest', help='Name of the Gemini model.')
    parser.add_argument('--api_key', type=str, default=None, help='Gemini API key.')

    # Tham số cho Few-shot (tùy chọn)
    parser.add_argument('--fewshot_vqa_file', type=str, default=None, help='(Few-shot) Path to the example VQA JSON file.')
    parser.add_argument('--fewshot_ht_file', type=str, default=None, help='(Few-shot) Path to the example Health Topic JSON file.')
    parser.add_argument('--fewshot_gt_file', type=str, default=None, help='(Few-shot) Path to the ground truth answers for examples.')
    parser.add_argument('--fewshot_image_dir', type=str, default=None, help='(Few-shot) Path to the example image directory.')
    parser.add_argument('--num_shots', type=int, default=1, help='(Few-shot) Number of examples to use in the prompt.')
    
    args = parser.parse_args()
    run_vqa_task(args)