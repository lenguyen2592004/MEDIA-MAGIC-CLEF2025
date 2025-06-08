import json
import argparse
from pathlib import Path
from tqdm import tqdm

def load_json(file_path: Path):
    """Tải và trả về dữ liệu từ một file JSON."""
    if not file_path.exists():
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_split(encounters_dict, answers_list, question_map, image_root_path, mode):
    """Xử lý một tập dữ liệu (train/val) để tạo các cặp image-question-answer."""
    processed_data = []
    print(f"Đang xử lý tập {mode}...")
    for item in tqdm(answers_list, desc=f"Processing {mode}"):
        encounter_id = item['encounter_id']
        encounter = encounters_dict.get(encounter_id)
        
        if not encounter or not encounter.get('image_ids'):
            continue

        # Tạo đường dẫn đến thư mục ảnh tương ứng (images_train hoặc images_valid)
        image_subfolder = image_root_path / f'images_{mode}'

        for qid, ans_idx in item.items():
            if qid == 'encounter_id' or qid not in question_map:
                continue

            q_details = question_map[qid]
            
            # Đảm bảo chỉ số câu trả lời hợp lệ
            if not isinstance(ans_idx, int) or ans_idx >= len(q_details['options']):
                continue
                
            answer_text = q_details['options'][ans_idx]

            for img_id in encounter['image_ids']:
                image_path = image_subfolder / img_id
                # Không cần kiểm tra exist ở đây để giữ cho code nhanh hơn,
                # nhưng giả định rằng các đường dẫn là chính xác.
                processed_data.append({
                    'qid': qid,
                    'encounter_id': encounter_id,
                    'image': str(image_path),
                    'question': q_details['question'],
                    'answer': answer_text,
                    'question_type': q_details['question_type_en'],
                    'answer_type': q_details['answer_type_en']
                })
    return processed_data

def process_test(encounters_list, question_map, image_root_path):
    """Xử lý tập test, tạo ra tất cả các cặp image-question có thể có."""
    test_data = []
    print("Đang xử lý tập test...")
    for encounter in tqdm(encounters_list, desc="Processing test"):
        encounter_id = encounter['encounter_id']
        image_ids = encounter.get('image_ids', [])
        
        if not image_ids:
            continue
            
        image_subfolder = image_root_path / 'images_test'

        for img_id in image_ids:
            image_path = image_subfolder / img_id
            for qid, q_details in question_map.items():
                test_data.append({
                    'qid': qid,
                    'encounter_id': encounter_id,
                    'image': str(image_path),
                    'question': q_details['question'],
                    'answer': "",  # Để trống câu trả lời cho tập test
                    'question_type': q_details['question_type_en'],
                    'answer_type': q_details['answer_type_en']
                })
    return test_data

def main(args):
    """Hàm chính để chạy toàn bộ quy trình."""
    
    # Chuyển đổi các chuỗi đường dẫn từ args thành đối tượng Path
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    # Xây dựng các đường dẫn file đầu vào
    encounters_file_train = base_dir / 'train_ht_v2.json'
    answers_file_train = base_dir / 'train_cvqa.json'
    encounters_file_val = base_dir / 'valid_ht_v2.json'
    answers_file_val = base_dir / 'valid_cvqa.json'
    encounters_file_test = base_dir / 'test_ht_v2.json'
    questions_file = base_dir / 'questions_v2.json'
    image_root = base_dir / 'images_final/images_final'

    # Tải tất cả dữ liệu
    print("Đang tải các file dữ liệu...")
    encounters_train = load_json(encounters_file_train)
    answers_train = load_json(answers_file_train)
    encounters_val = load_json(encounters_file_val)
    answers_val = load_json(answers_file_val)
    encounters_test = load_json(encounters_file_test)
    questions = load_json(questions_file)

    if any(data is None for data in [encounters_train, answers_train, encounters_val, answers_val, encounters_test, questions]):
        print("Một hoặc nhiều file đầu vào không thể tải được. Đang thoát.")
        return

    # Tối ưu hóa: Chuyển đổi danh sách thành dict để truy cập nhanh
    encounters_train_dict = {e['encounter_id']: e for e in encounters_train}
    encounters_val_dict = {e['encounter_id']: e for e in encounters_val}

    # Tạo map câu hỏi
    question_map = {
        q['qid']: {
            'question': q['question_en'],
            'options': q['options_en'],
            'question_type_en': q['question_type_en'],
            'question_category_en': q['question_category_en']
        } for q in questions
    }
    
    # Xử lý các tập dữ liệu
    train_data = process_split(encounters_train_dict, answers_train, question_map, image_root, 'train')
    val_data = process_split(encounters_val_dict, answers_val, question_map, image_root, 'valid')
    test_data = process_test(encounters_test, question_map, image_root)
    
    # Tạo danh sách các đáp án duy nhất
    print("Đang tạo danh sách câu trả lời...")
    answer_list = sorted(list(set(item['answer'] for item in train_data + val_data)))
    
    # Ghi file
    print("Đang ghi các file đầu ra...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'answer_list.json', 'w', encoding='utf-8') as f:
        json.dump(answer_list, f, ensure_ascii=False, indent=2)

    print(f"✅ Hoàn thành! Đã tạo các file trong thư mục: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu VQA từ các file nguồn.")
    
    parser.add_argument(
        '--base_dir', 
        type=str, 
        required=True,
        help='(Bắt buộc) Đường dẫn đến thư mục gốc chứa tất cả các file JSON và thư mục ảnh.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='vqa_processed_data',
        help='Thư mục để lưu các file đã xử lý. (Mặc định: vqa_processed_data)'
    )

    args = parser.parse_args()
    main(args)