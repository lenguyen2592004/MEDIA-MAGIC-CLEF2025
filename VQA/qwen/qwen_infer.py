# -*- coding: utf-8 -*-
"""ImageCLEF VQA inference using Qwen 2.5-VL model

This script performs Visual Question Answering for dermatology images using the Qwen 2.5-VL model.
It supports voting mechanism and few-shot learning for improved accuracy.
"""

import json
import os
import random
import argparse
import re
import gc
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import gc
from collections import defaultdict, Counter
import traceback
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

# Parse command line arguments
parser = argparse.ArgumentParser(description="VQA using Qwen 2.5-VL 2B Instruct")
parser.add_argument('--val_vqa_dataset', type=str, required=True, help='Path to val_vqa_dataset.json')
parser.add_argument('--valid_ht_v2', type=str, required=True, help='Path to valid_ht_v2.json')
parser.add_argument('--train_vqa_dataset', type=str, required=True, help='Path to train_vqa_dataset.json')
parser.add_argument('--train_json', type=str, required=True, help='Path to train.json')
parser.add_argument('--train_cvqa', type=str, required=True, help='Path to train_cvqa.json')
parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
parser.add_argument('--task', type=str, required=True, choices=['train', 'valid', 'test','combined_enhanced_images'], help='Task type')
parser.add_argument('--output_file', type=str, default='prediction.json', help='Output file path')
parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
args = parser.parse_args()

# Function to clear GPU memory cache
def clear_gpu_cache():
    torch.cuda.empty_cache()
    gc.collect()

# Load the model and processor
model_name = "Qwen/Qwen2-VL-7B-Instruct"
# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Create image processor with required parameters
image_processor = Qwen2VLImageProcessor(
    size={"shortest_edge": 480, "longest_edge": 640},
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)

# Load tokenizer and combine with image processor
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True  # Enable fast tokenizer
)

processor = AutoProcessor.from_pretrained(
    model_name,
    tokenizer=tokenizer,
    image_processor=image_processor,
    quantization_config=quantization_config,
    trust_remote_code=True
)

# Configure GPU memory usage based on available GPUs
gpu_memory = {}
if args.num_gpus > 0:
    per_gpu_mem = "12GiB"  # Adjust based on your GPU memory
    for i in range(args.num_gpus):
        gpu_memory[i] = per_gpu_mem

    print(f"Using {args.num_gpus} GPUs with memory configuration: {gpu_memory}")
else:
    print("No GPUs specified, using CPU only")

print('CHECKPOINT1\n')

# Load model with quantization
try:
    print('CHECKPOINT2\n')
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        # max_memory=gpu_memory,  # This parameter is missing
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    model.eval()
    # Explicitly move model to device to ensure quantization layers are initialized
    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed 'unsloth': pip install unsloth")
    exit(1)
except RuntimeError as e:
    print(f"Runtime error: {e}")
    print("Try using fewer GPUs or a smaller model")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)
print('CHECKPOINT3\n')

# Load data files
print("Loading data files...")
with open(args.val_vqa_dataset, 'r', encoding='utf-8') as f:
    val_vqa_dataset = json.load(f)
print('CHECKPOINT4\n')

with open(args.valid_ht_v2, 'r', encoding='utf-8') as f:
    valid_ht_v2 = json.load(f)

with open(args.train_vqa_dataset, 'r', encoding='utf-8') as f:
    train_vqa_dataset = json.load(f)
print('CHECKPOINT5\n')
with open(args.train_json, 'r', encoding='utf-8') as f:
    train_json = json.load(f)

with open(args.train_cvqa, 'r', encoding='utf-8') as f:
    train_cvqa = json.load(f)
def shuffle_answers_with_indices(answers):
    """
    Shuffle answers while preserving their original indices.
    Returns shuffled answers with their original indices.
    """
    indexed_answers = list(enumerate(answers))
    random.shuffle(indexed_answers)
    return indexed_answers
# Create dictionaries for faster access
encounter_to_images = {item['encounter_id']: item['image_ids'] for item in valid_ht_v2 if 'encounter_id' in item and 'image_ids' in item}
encounter_to_caption = {item['encounter_id']: item.get('query_content_en', '') for item in valid_ht_v2 if 'encounter_id' in item}

# Organize val_vqa_dataset by encounter_id
encounter_to_qids = {}
for item in val_vqa_dataset:
    encounter_id = item.get('encounter_id')
    if not encounter_id:
        continue
    if encounter_id not in encounter_to_qids:
        encounter_to_qids[encounter_id] = []
    encounter_to_qids[encounter_id].append(item['qid'])

# Organize val_vqa_dataset by qid
qid_to_data = {item['qid']: item for item in val_vqa_dataset if 'qid' in item}
# 1. Thêm danh sách CQID bắt buộc và khởi tạo giá trị mặc định
REQUIRED_CQIDS = [
    "CQID010-001", "CQID011-001", "CQID011-002", "CQID011-003",
    "CQID011-004", "CQID011-005", "CQID011-006", "CQID012-001",
    "CQID012-002", "CQID012-003", "CQID012-004", "CQID012-005",
    "CQID012-006", "CQID015-001", "CQID020-001", "CQID020-002",
    "CQID020-003", "CQID020-004", "CQID020-005", "CQID020-006",
    "CQID020-007", "CQID020-008", "CQID020-009", "CQID025-001",
    "CQID034-001", "CQID035-001", "CQID036-001"
]

# Function to get few-shot examples from training data
def get_few_shot_examples():
    print("Preparing few-shot examples...")
    encounter_to_answers = {}
    for item in train_cvqa:
        if 'encounter_id' not in item:
            continue
        encounter_id = item['encounter_id']
        encounter_to_answers[encounter_id] = {k: v for k, v in item.items() if k != 'encounter_id'}

    train_encounter_to_caption = {item['encounter_id']: item.get('query_content_en', '')
                                 for item in train_json if 'encounter_id' in item}

    train_encounter_to_qids = {}
    for item in train_vqa_dataset:
        if 'encounter_id' not in item:
            continue
        encounter_id = item['encounter_id']
        if encounter_id not in train_encounter_to_qids:
            train_encounter_to_qids[encounter_id] = []
        train_encounter_to_qids[encounter_id].append(item)

    valid_encounters = [enc for enc in train_encounter_to_qids.keys() if enc in encounter_to_answers]
    if not valid_encounters:
        print("No valid training examples found!")
        return []

    random_encounters = random.sample(valid_encounters, min(2, len(valid_encounters)))

    examples = []
    for encounter_id in random_encounters:
        if encounter_id in train_encounter_to_qids:
            questions = train_encounter_to_qids[encounter_id]
            valid_questions = [q for q in questions if q['qid'] in encounter_to_answers.get(encounter_id, {})]
            if not valid_questions:
                continue

            random_question = random.choice(valid_questions)
            qid = random_question['qid']

            answer_idx = encounter_to_answers[encounter_id].get(qid)
            if answer_idx is None:
                continue

            examples.append({
                'qid': qid,
                'encounter_id': encounter_id,
                'question': random_question['question'],
                'answers': random_question['answer'],
                'answer_idx': answer_idx,
                'caption': train_encounter_to_caption.get(encounter_id, '')
            })

    print(f"Found {len(examples)} few-shot examples")
    return examples[:2]

def get_few_shot_examples_ensure_required(train_cvqa, train_json, train_vqa_dataset):
    """
    Tạo các ví dụ few-shot, đảm bảo mỗi QID trong REQUIRED_CQIDS
    được đại diện ít nhất một lần nếu có thể.
    """
    print("Preparing few-shot examples with required QIDs coverage...")

    # --- Bước 1: Chuẩn bị các cấu trúc dữ liệu ánh xạ ---
    encounter_to_answers = {}
    for item in train_cvqa:
        if 'encounter_id' not in item:
            continue
        encounter_id = item['encounter_id']
        encounter_to_answers[encounter_id] = {k: v for k, v in item.items() if k != 'encounter_id'}

    train_encounter_to_caption = {item['encounter_id']: item.get('query_content_en', '')
                                 for item in train_json if 'encounter_id' in item}

    train_encounter_to_questions = defaultdict(list) # Sử dụng defaultdict cho tiện
    for item in train_vqa_dataset:
        if 'encounter_id' not in item:
            continue
        encounter_id = item['encounter_id']
        train_encounter_to_questions[encounter_id].append(item)

    # --- Bước 2: Xác định các encounter hợp lệ (có cả câu hỏi và câu trả lời) ---
    valid_encounters = [
        enc for enc in train_encounter_to_questions.keys() if enc in encounter_to_answers
    ]
    if not valid_encounters:
        print("No valid training encounters found (having both questions and answers)!")
        return []

    # --- Bước 3: Tạo danh sách tất cả các ví dụ hợp lệ có QID nằm trong REQUIRED_CQIDS ---
    possible_examples_by_qid = defaultdict(list)
    print(f"Scanning {len(valid_encounters)} valid encounters for required QIDs...")

    for encounter_id in valid_encounters:
        questions = train_encounter_to_questions[encounter_id]
        answers_for_encounter = encounter_to_answers.get(encounter_id, {})

        for question_data in questions:
            qid = question_data.get('qid')
            if not qid:
                continue # Bỏ qua nếu câu hỏi không có qid

            # Chỉ xem xét các QID bắt buộc
            if qid in REQUIRED_CQIDS:
                # Kiểm tra xem có câu trả lời cho QID này không
                answer_idx = answers_for_encounter.get(qid)
                if answer_idx is not None:
                    # Tạo ví dụ tiềm năng
                    example = {
                        'qid': qid,
                        'encounter_id': encounter_id,
                        'question': question_data.get('question', ''),
                        'answers': question_data.get('answer', []), # Đổi tên 'answer' thành 'answers' nếu nó là list các lựa chọn
                        'answer_idx': answer_idx,
                        'caption': train_encounter_to_caption.get(encounter_id, '')
                    }
                    possible_examples_by_qid[qid].append(example)

    # --- Bước 4: Chọn ngẫu nhiên một ví dụ cho mỗi REQUIRED_CQID tìm thấy ---
    final_examples = []
    covered_qids = set()
    required_qids_set = set(REQUIRED_CQIDS)

    print("Selecting one example for each required QID found...")
    # Ưu tiên chọn từ các QID bắt buộc
    for qid in REQUIRED_CQIDS:
        if qid in possible_examples_by_qid and possible_examples_by_qid[qid]:
            # Chọn ngẫu nhiên 1 ví dụ từ các ví dụ có sẵn cho qid này
            chosen_example = random.choice(possible_examples_by_qid[qid])
            final_examples.append(chosen_example)
            covered_qids.add(qid)
        else:
            print(f"Warning: No valid example found for required QID: {qid}")

    print(f"Found examples covering {len(covered_qids)} out of {len(required_qids_set)} required QIDs.")
    print(f"Total examples selected: {len(final_examples)}")

    # Trả về các ví dụ đã chọn
    return final_examples

# Get few-shot examples
few_shot_examples = get_few_shot_examples_ensure_required(train_cvqa, train_json, train_vqa_dataset)
import re  # Ensure 're' is imported at the top of the script


def process_with_voting(model, processor, image_ids_path, question, answers,
                        caption, qid, encounter_id, max_attempts=1,
                        tiebreaker_attempts=3, candidates=None):
    """
    Process image using voting mechanism with shuffled answers.
    If candidates is provided, only those indices will be considered.
    """
    try:
        print(f"Starting voting process for question {qid} with {max_attempts} attempts")

        # Track votes for each answer index
        votes = Counter()

        # Process multiple times with different shuffled orders
        for attempt in range(max_attempts):
            # If we have candidates, only use those; otherwise use all answers
            if candidates:
                indexed_candidates = [(idx, answers[idx]) for idx in candidates]
                print(f"indexed_candidates: {indexed_candidates}\n")
                shuffled_answers = shuffle_answers_with_indices(indexed_candidates)
                print(f"shuffled_answers: {shuffled_answers}\n")
                # Create mapping from position to original index
                position_to_index = {i: idx for i, (idx, _) in enumerate(shuffled_answers)}
                print(f"position_to_index: {position_to_index}\n")
                # Extract just the answers for the prompt
                shuffled_answer_texts = [ans for _, ans in shuffled_answers]
                print(f"shuffled_answer_texts: {shuffled_answer_texts}\n")
            else:
                # Shuffle all answers
                # len_ans_goc=len(answers)
                shuffled_answers = shuffle_answers_with_indices(answers)
                print(f"shuffled_answer_texts: {shuffled_answers}\n")
                position_to_index = {i: idx for i, (idx, _) in enumerate(shuffled_answers)}
                print(f"position_to_index: {position_to_index}\n")
                shuffled_answer_texts = [ans for _, ans in shuffled_answers]
                print(f"shuffled_answer_texts: {shuffled_answer_texts}\n")

            # Build content with images
            content = []
            for img_path in image_ids_path:
                content.append({'type': 'image', 'path': img_path})

            if not content:
                return "No images"

            # Build prompt with shuffled answers
            prompt = f"""You are a specialized medical visual question answering assistant focusing on dermatology. Your expertise lies in analyzing dermatological images and related textual information to provide accurate answers.

**Task:** Analyze the provided dermatology image(s) and the accompanying image caption to answer the multiple-choice question. Select the single best answer from the provided options and output *only* its 0-based index.

**Context:**
*   **Encounter ID:** {encounter_id}
*   **Image(s):** [Image provided separately]
*   **Image Caption:** {caption}

**Question (qid: {qid}):**
{question}

**Answer Options (Select one index):**
"""

            # Add shuffled answer options
            for i, answer in enumerate(shuffled_answer_texts):
                prompt += f"{i}: {answer}\n"

            prompt += f"""
**Instructions:**
1.  Carefully examine the provided images. Pay attention to visual details relevant to dermatological conditions (lesions, patterns, colors, textures, distribution, location).
2.  Read the **Image Caption** carefully as it may contain crucial supplementary information about the case or images.
3.  Understand the specific **Question** being asked.
4.  Evaluate each **Answer Option** based *solely* on the visual evidence in the image(s) and the information in the caption.
5.  Determine the **single most accurate** answer from the provided list.
6.  Output ONLY the numerical index (0, 1, 2, etc.) at the first word of your selected answer.

**Output only the index number:**
There are some ground truth examples from other dataset to help this mission easier.
----------FEW SHOT EXAMPLES START----------------------------------------------\n
{few_shot_examples}\n
----------FEW SHOT EXAMPLES END----------------------------------------------\n
"""

            # Add prompt to content
            content.append({'type': 'text', 'text': prompt})

            # Create conversation
            conversation = [{"role": "user", "content": content}]

            # Get model output
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate response
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            # Extract response
            input_token_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_token_len:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            # Xóa các tensor và giải phóng bộ nhớ
            del inputs, output_ids, generated_ids
            clear_gpu_cache() # Gọi hàm đã định nghĩa để dọn dẹp cả GPU và CPU RAM
            # Process numeric response
            try:
                # Extract first number from response
                position = int(response.strip())
                # Convert position in shuffled order to original index
                # original_idx = position_to_index.get(position, -1)
                original_idx = position
                if original_idx != -1:  # Valid response
                    votes[original_idx] += 1
                    print(f"  Attempt {attempt+1}: Position {position} maps to original index {original_idx}")
                else:
                    print(f"  Attempt {attempt+1}: Invalid position {position}, ignoring")
            except (ValueError, TypeError):
                print(f"  Attempt {attempt+1}: Non-numeric response: '{response}'")

        # Find most common vote(s)
        if not votes:
            print("No valid votes collected!")
            return -1

        most_common = votes.most_common()
        most_votes = most_common[0][1]
        top_candidates = [idx for idx, count in most_common if count == most_votes]

        print(f"Vote results: {dict(votes)}")
        print(f"Top candidates: {top_candidates} with {most_votes} votes each")

        # If we have a clear winner, return it
        if len(top_candidates) == 1:
            return top_candidates[0]

        # If we have a tie and this isn't already a tiebreaker round
        if len(top_candidates) > 1 and candidates is None:
            print(f"Tie detected among indices {top_candidates}. Running tiebreaker...")
            # Run tiebreaker with only the tied candidates
            return process_with_voting(
                model, processor, image_ids_path, question, answers,
                caption, qid, encounter_id, max_attempts=tiebreaker_attempts,
                candidates=top_candidates
            )

        # If we still have a tie after tiebreaker, just pick the first one
        return top_candidates[0]

    except Exception as e:
        print(f"Error in voting process: {e}")
        traceback.print_exc()
        return -1
# Process each encounter and answer questions
results = []
total_encounters = len(valid_ht_v2)
print('CHECKPOINT')
for i, encounter_data in enumerate(valid_ht_v2):
    if 'encounter_id' not in encounter_data:
        continue
    if i>=54:
        # NHỚ XÓA
        encounter_id = encounter_data['encounter_id']
        image_ids = encounter_data.get('image_ids', [])
        caption = encounter_data.get('query_content_en', '')

        print(f"\nProcessing encounter {i+1}/{total_encounters}: {encounter_id}")

        qids = encounter_to_qids.get(encounter_id, [])
        if not qids:
            print(f"No questions found for encounter {encounter_id}")
            continue

        encounter_result = {"encounter_id": encounter_id}
        count_question = 0
        len_qid = len(qids)

        # clear_gpu_cache()

        image_ids_path = []
        if image_ids:
            for image_id in image_ids:
                img_path = os.path.join(args.image_dir, f"images_{args.task}/{image_id}")
                if os.path.exists(img_path):
                    image_ids_path.append(img_path)
        for index in range(0,len_qid):
            # if index!=13:
            #     continue
            # NHỚ XÓA
            if index >=0:
                qid=qids[index]
                question_data = qid_to_data.get(qid)
                if not question_data:
                    print(f"No data found for qid {qid}")
                    continue
                question = question_data['question']
                answers = question_data['answer']

                print(f"Processing question {count_question + 1}/{len_qid}: {qid}")

                # max_attempts = 2
                answer_idx = -1
                # clear_gpu_cache()
                # for attempt in range(max_attempts):
                try:
                    # response = process_image(
                    #     model, processor, image_ids_path, question, answers, caption, qid, encounter_id
                    # )
                    # Use voting process instead of single attempt
                    answer_idx = process_with_voting(
                        model, processor, image_ids_path, question,
                        answers, caption, qid, encounter_id
                    )
                    # answer_idx=response[0]
                    # print(f"  Attempt {attempt+1}: Model response: '{answer_idx}' | Selected index: {answer_idx}")
                    print(f"  Model response: '{answer_idx}' | Selected index: {answer_idx}")
                except Exception as e:
                    print(f"  Attempt failed: {e}")
                encounter_result.update({qid: answer_idx})
                # encounter_result
                print(f"qid: {qid}")
                # encounter_result+=
                # encounter_result[qid] = answer_idx
                count_question += 1

                if count_question % 1 == 0:
                    with open(f"{args.output_file}.partial", 'w', encoding='utf-8') as f:
                        partial_results = results + [encounter_result]
                        print(f"PARTIAL {partial_results}")
                        json.dump(partial_results, f, indent=4)
        print(encounter_result)
        results.append(encounter_result)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        torch.cuda.empty_cache()
        gc.collect()

print(f"\nPredictions saved to {args.output_file}")