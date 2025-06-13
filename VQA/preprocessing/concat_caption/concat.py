import json
import os
import argparse # Thêm thư viện argparse

def merge_json_captions_replace_newline(json_file_paths, output_file_path):
    """
    Gộp caption từ nhiều file JSON vào một file JSON mới,
    thay thế ký tự xuống dòng ('\\n') bằng dấu chấm ('.') trong mỗi caption.

    Args:
        json_file_paths (list): Danh sách các đường dẫn đến file JSON đầu vào.
                                File đầu tiên trong danh sách sẽ quyết định cấu trúc
                                (các key và trường 'image').
        output_file_path (str): Đường dẫn để lưu file JSON đã gộp.
    """
    if not json_file_paths:
        print("Lỗi: Không có đường dẫn file JSON đầu vào nào được cung cấp.")
        return

    merged_data = {}
    first_file_path = json_file_paths[0]

    # --- Bước 1: Đọc file đầu tiên để lấy cấu trúc cơ bản và caption đầu tiên ---
    print(f"Đang xử lý file cơ sở: {first_file_path}")
    try:
        with open(first_file_path, 'r', encoding='utf-8') as f:
            first_file_data = json.load(f)

        for image_id, data in first_file_data.items():
            image_path = data.get('image')
            raw_caption = data.get('caption', '')
            cleaned_initial_caption = raw_caption.replace('\n', '.').strip()

            if image_path is not None:
                merged_data[image_id] = {
                    'image': image_path,
                    'caption': cleaned_initial_caption
                }
            else:
                print(f"Cảnh báo: Thiếu trường 'image' cho ID '{image_id}' trong file cơ sở {first_file_path}. Bỏ qua.")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file cơ sở: {first_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ file cơ sở: {first_file_path}")
        return
    except Exception as e:
        print(f"Lỗi không mong muốn xảy ra khi xử lý file cơ sở {first_file_path}: {e}")
        return

    # --- Bước 2: Đọc các file còn lại và nối caption đã được làm sạch ---
    for file_path in json_file_paths[1:]:
        print(f"Đang xử lý file bổ sung: {file_path}")
        try:
            if not os.path.exists(file_path):
                print(f"Cảnh báo: Không tìm thấy file, bỏ qua: {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)

            for image_id, data in current_data.items():
                if image_id in merged_data:
                    raw_caption = data.get('caption', '')
                    cleaned_additional_caption = raw_caption.replace('\n', '.').strip()

                    if cleaned_additional_caption:
                        merged_data[image_id]['caption'] += " " + cleaned_additional_caption

        except json.JSONDecodeError:
            print(f"Cảnh báo: Không thể giải mã JSON từ file: {file_path}. Bỏ qua file này.")
        except Exception as e:
            print(f"Lỗi không mong muốn xảy ra khi xử lý file {file_path}: {e}. Bỏ qua file này.")

    # --- Bước 3: Ghi kết quả ra file mới ---
    print(f"Đang ghi dữ liệu đã gộp vào: {output_file_path}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        print("Đã gộp caption thành công (ký tự xuống dòng được thay bằng dấu chấm)!")

    except IOError as e:
        print(f"Lỗi: Không thể ghi vào file đầu ra {output_file_path}: {e}")
    except Exception as e:
        print(f"Lỗi không mong muốn xảy ra khi ghi file đầu ra: {e}")

# --- Cách sử dụng mới với argparse ---
def main():
    # Tạo một parser để xử lý các đối số dòng lệnh
    parser = argparse.ArgumentParser(
        description="Gộp nhiều file caption JSON thành một. File đầu vào đầu tiên được dùng làm file cơ sở.",
        formatter_class=argparse.RawTextHelpFormatter # Giúp hiển thị help text đẹp hơn
    )

    # Thêm đối số cho các file đầu vào (input)
    # nargs='+' có nghĩa là nhận một hoặc nhiều giá trị
    parser.add_argument(
        'input_files',
        metavar='INPUT_FILE',
        nargs='+',
        help='Danh sách các đường dẫn đến file JSON đầu vào.\n'
             'File đầu tiên sẽ được sử dụng làm cơ sở để lấy cấu trúc và đường dẫn ảnh.'
    )

    # Thêm đối số cho file đầu ra (output)
    # -o là viết tắt, --output là viết đầy đủ. Bắt buộc phải có (required=True)
    parser.add_argument(
        '-o', '--output',
        metavar='OUTPUT_FILE',
        required=True,
        help='Đường dẫn đến file JSON đầu ra để lưu kết quả.'
    )

    # Phân tích các đối số được truyền vào từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm xử lý chính với các đối số đã được phân tích
    merge_json_captions_replace_newline(args.input_files, args.output)

if __name__ == "__main__":
    main()
