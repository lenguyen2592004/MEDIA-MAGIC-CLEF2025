import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def create_pretraining_data(image_dir: Path, input_json_path: Path, output_json_path: Path):
    """
    Tạo dữ liệu pre-training (image-caption pairs) từ file valid_ht_v2.json.

    Args:
        image_dir (Path): Đường dẫn đến thư mục chứa các file ảnh.
        input_json_path (Path): Đường dẫn đến file JSON đầu vào (valid_ht_v2.json).
        output_json_path (Path): Đường dẫn để lưu file JSON đầu ra.
    """
    print("Starting pre-training data creation...")
    print(f"Image directory: {image_dir}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output JSON: {output_json_path}")

    # Kiểm tra xem file input có tồn tại không
    if not input_json_path.exists():
        print(f"Error: Input file not found at {input_json_path}")
        return
        
    if not image_dir.is_dir():
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Load data
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    pretrain_data = []

    print(f"\nProcessing {len(data)} items...")
    for item in tqdm(data, desc="Processing items"):
        image_ids = item.get("image_ids", [])
        # Lấy nội dung tiếng Anh làm caption
        query_content_en = item.get("query_content_en", "").strip()

        # Bỏ qua nếu không có caption hoặc không có ảnh
        if not query_content_en or not image_ids:
            continue

        for img_name in image_ids:
            # Sử dụng pathlib để nối đường dẫn một cách an toàn
            img_path = image_dir / img_name
            
            # Chỉ thêm vào nếu file ảnh thực sự tồn tại
            if img_path.exists():
                pretrain_data.append({
                    "image": str(img_path),  # Lưu đường dẫn dưới dạng chuỗi
                    "caption": query_content_en
                })
            else:
                print(f"Warning: Image file not found and will be skipped: {img_path}")


    # Save file json
    print(f"\nSaving {len(pretrain_data)} pre-training samples to {output_json_path}...")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(pretrain_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully created pre-training data at: {output_json_path}")
    except Exception as e:
        print(f"Error: Failed to save output file. {e}")


if __name__ == "__main__":
    # Tạo parser cho các đối số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Create image-caption pre-training data from the VQA-Med-2025 dataset."
    )
    
    # Định nghĩa các đối số
    parser.add_argument(
        '--image_dir', 
        type=str, 
        required=True,
        help='(Required) Path to the directory containing the validation images.'
    )
    parser.add_argument(
        '--input_json', 
        type=str, 
        required=True,
        help='(Required) Path to the input JSON file (e.g., valid_ht_v2.json).'
    )
    parser.add_argument(
        '--output_json', 
        type=str, 
        default='pretrain_data_val.json',
        help='Path to save the generated pre-training JSON file. (default: pretrain_data_val.json)'
    )

    # Parse các đối số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm chính với các đối số đã được parse
    create_pretraining_data(
        image_dir=Path(args.image_dir),
        input_json_path=Path(args.input_json),
        output_json_path=Path(args.output_json)
    )