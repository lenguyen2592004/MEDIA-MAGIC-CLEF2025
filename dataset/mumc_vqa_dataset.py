import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

def load_json(file_path: Path):
    """Tải và trả về dữ liệu từ một file JSON."""
    if not file_path.exists():
        print(f"Lỗi: Không tìm thấy file {file_path}", file=sys.stderr)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Lỗi: File JSON không hợp lệ {file_path}. Chi tiết: {e}", file=sys.stderr)
        return None

def process_split(encounters_dict, answers_list, question_map, image_folder_path: Path, mode: str):
    """Xử lý một tập dữ liệu (train/val) để tạo các cặp image-question-answer."""
    processed_data = []
    print(f"\nĐang xử lý tập {mode} với ảnh từ: {image_folder_path}")
    if not image_folder_path.is_dir():
        print(f"⚠️ Cảnh báo: Thư mục ảnh {image_folder_path} không tồn tại hoặc không phải là thư mục.", file=sys.stderr)
    
    for item in tqdm(answers_list, desc=f"Processing {mode}"):
        encounter_id = item['encounter_id']
        encounter = encounters_dict.get(encounter_id)
        
        if not encounter or not encounter.get('image_ids'):
            continue

        for qid, ans_idx in item.items():
            if qid == 'encounter_id' or qid not in question_map:
                continue

            q_details = question_map[qid]
            
            if not isinstance(ans_idx, int) or ans_idx >= len(q_details['options']):
                continue
                
            answer_text = q_details['options'][ans_idx]

            for img_id in encounter['image_ids']:
                image_path = image_folder_path / img_id
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

def process_test(encounters_list, question_map, test_image_folder_path: Path):
    """Xử lý tập test, tạo ra tất cả các cặp image-question có thể có."""
    test_data = []
    print(f"\nĐang xử lý tập test với ảnh từ: {test_image_folder_path}")
    if not test_image_folder_path.is_dir():
        print(f"⚠️ Cảnh báo: Thư mục ảnh {test_image_folder_path} không tồn tại hoặc không phải là thư mục.", file=sys.stderr)
        
    for encounter in tqdm(encounters_list, desc="Processing test"):
        encounter_id = encounter['encounter_id']
        image_ids = encounter.get('image_ids', [])
        
        if not image_ids:
            continue
            
        for img_id in image_ids:
            image_path = test_image_folder_path / img_id
            for qid, q_details in question_map.items():
                test_data.append({
                    'qid': qid,
                    'encounter_id': encounter_id,
                    'image': str(image_path),
                    'question': q_details['question'],
                    'answer': "",
                    'question_type': q_details['question_type_en'],
                    'answer_type': q_details['answer_type_en']
                })
    return test_data

def main(args):
    """Hàm chính để chạy toàn bộ quy trình chuẩn bị dữ liệu."""
    
    # Chuyển đổi tất cả các chuỗi đường dẫn đầu vào thành đối tượng Path
    file_paths = {
        'train_encounters': Path(args.train_encounters_json),
        'train_answers': Path(args.train_answers_json),
        'valid_encounters': Path(args.valid_encounters_json),
        'valid_answers': Path(args.valid_answers_json),
        'test_encounters': Path(args.test_encounters_json),
        'questions': Path(args.questions_json),
    }
    image_dirs = {
        'train': Path(args.train_image_dir),
        'valid': Path(args.valid_image_dir),
        'test': Path(args.test_image_dir),
    }
    output_dir = Path(args.output_dir)

    print("--- Tóm tắt cấu hình ---")
    for name, path in file_paths.items():
        print(f"{name.replace('_', ' ').title():<20}: {path}")
    for name, path in image_dirs.items():
        print(f"{name.title()} Image Directory : {path}")
    print(f"{'Output Directory':<20}: {output_dir}")
    print("--------------------------")
    
    # Tải tất cả dữ liệu
    print("\nĐang tải các file dữ liệu...")
    loaded_data = {name: load_json(path) for name, path in file_paths.items()}

    if any(data is None for data in loaded_data.values()):
        print("\nLỗi: Một hoặc nhiều file đầu vào không thể tải được. Vui lòng kiểm tra lại các đường dẫn. Đang thoát.", file=sys.stderr)
        sys.exit(1)

    encounters_train_dict = {e['encounter_id']: e for e in loaded_data['train_encounters']}
    encounters_val_dict = {e['encounter_id']: e for e in loaded_data['valid_encounters']}

    question_map = {
        q['qid']: {
            'question': q['question_en'],
            'options': q['options_en'],
            'question_type_en': q['question_type_en'],
            'answer_type_en': q.get('answer_type_en', 'unknown')
        } for q in loaded_data['questions']
    }
    
    # Xử lý các tập dữ liệu
    train_data = process_split(encounters_train_dict, loaded_data['train_answers'], question_map, image_dirs['train'], 'train')
    val_data = process_split(encounters_val_dict, loaded_data['valid_answers'], question_map, image_dirs['valid'], 'valid')
    test_data = process_test(loaded_data['test_encounters'], question_map, image_dirs['test'])
    
    # Tạo danh sách các đáp án duy nhất
    print("\nĐang tạo danh sách câu trả lời...")
    all_answers = set(item['answer'] for item in train_data + val_data if item['answer'])
    answer_list = sorted(list(all_answers))
    
    # Ghi file
    print(f"\nĐang ghi các file đầu ra vào thư mục: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {'train.json': train_data, 'val.json': val_data, 'test.json': test_data, 'answer_list.json': answer_list}
    
    for filename, data in output_files.items():
        try:
            with open(output_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Lỗi khi ghi file {filename}: {e}", file=sys.stderr)

    print(f"\n✅ Hoàn thành! Đã tạo {len(output_files)} file trong thư mục: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu VQA, cho phép chỉ định đường dẫn độc lập cho từng file đầu vào.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Đối số cho các file JSON ---
    parser.add_argument('--train_encounters_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file encounter của tập train (train_ht_v2.json).")
    parser.add_argument('--train_answers_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file câu trả lời của tập train (train_cvqa.json).")
    parser.add_argument('--valid_encounters_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file encounter của tập validation (valid_ht_v2.json).")
    parser.add_argument('--valid_answers_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file câu trả lời của tập validation (valid_cvqa.json).")
    parser.add_argument('--test_encounters_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file encounter của tập test (test_ht_v2.json).")
    parser.add_argument('--questions_json', type=str, required=True, help="(Bắt buộc) Đường dẫn đến file chứa danh sách câu hỏi (questions_v2.json).")

    # --- Đối số cho các thư mục ảnh ---
    parser.add_argument('--train_image_dir', type=str, required=True, help="(Bắt buộc) Đường dẫn đến thư mục chứa ảnh train.")
    parser.add_argument('--valid_image_dir', type=str, required=True, help="(Bắt buộc) Đường dẫn đến thư mục chứa ảnh valid.")
    parser.add_argument('--test_image_dir', type=str, required=True, help="(Bắt buộc) Đường dẫn đến thư mục chứa ảnh test.")

    # --- Đối số cho thư mục đầu ra ---
    parser.add_argument('--output_dir', type=str, default='vqa_processed_data', help="Thư mục để lưu các file đã xử lý.")

    args = parser.parse_args()
    main(args)
