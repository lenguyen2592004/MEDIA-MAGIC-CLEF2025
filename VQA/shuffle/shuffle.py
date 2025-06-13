import json
import random
import argparse  

def shuffle_answers_in_data(dataset):
    """
    Xáo trộn danh sách 'answer' cho mỗi câu hỏi trong một dataset (đối tượng Python).

    Args:
        dataset (list): Một danh sách các dictionary, mỗi dictionary là một mục dữ liệu.

    Returns:
        list: Dataset đã được sửa đổi với các câu trả lời đã được xáo trộn.
    """
    # Kiểm tra xem đầu vào có phải là một list không
    if not isinstance(dataset, list):
        print("Cảnh báo: Dữ liệu đầu vào không phải là một danh sách. Không thực hiện xáo trộn.")
        return dataset

    for item in dataset:
        # Kiểm tra xem 'answer' có tồn tại và là một list không
        if "answer" in item and isinstance(item["answer"], list):
            # random.shuffle() thực hiện xáo trộn ngay trên list đó
            random.shuffle(item["answer"])

    return dataset

def main(args):
    """
    Hàm chính điều khiển luồng công việc: đọc, xử lý, và ghi file.
    """
    input_path = args.input
    output_path = args.output

    print(f"Đang đọc dữ liệu từ: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # Sử dụng json.load() để chuyển đổi trực tiếp từ file thành đối tượng Python
            data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{input_path}'")
        return  # Dừng chương trình nếu file không tồn tại
    except json.JSONDecodeError as e:
        print(f"Lỗi: Không thể giải mã JSON từ file '{input_path}'. Lỗi: {e}")
        return

    print("Đang xáo trộn các câu trả lời...")
    shuffled_data = shuffle_answers_in_data(data)
    print("Xáo trộn hoàn tất.")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Sử dụng json.dump() để ghi đối tượng Python vào file JSON
            json.dump(shuffled_data, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu thành công dữ liệu đã xáo trộn vào: {output_path}")
    except IOError as e:
        print(f"Lỗi: Không thể ghi vào file '{output_path}'. Lỗi: {e}")

# --- Khối thực thi chính ---
if __name__ == "__main__":
    # 1. Tạo một đối tượng ArgumentParser
    parser = argparse.ArgumentParser(
        description="Xáo trộn danh sách 'answer' trong một file JSON dataset.",
        epilog="Ví dụ: python ten_script.py -i data.json -o shuffled_data.json"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,  
        help="Đường dẫn đến file JSON đầu vào."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,  
        help="Đường dẫn để lưu file JSON đầu ra đã được xáo trộn."
    )

    args = parser.parse_args()

    main(args)