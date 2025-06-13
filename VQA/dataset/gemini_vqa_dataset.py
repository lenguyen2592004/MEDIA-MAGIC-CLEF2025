import json
import os
import argparse  # Thêm thư viện argparse
import traceback

def create_vqa_dataset(question_file, definitions_file, output_file):
    """
    Kết hợp dữ liệu câu hỏi (encounter, image_ids) với dữ liệu định nghĩa 
    (qid, question, answer) để tạo ra một bộ dữ liệu VQA hoàn chỉnh.

    Args:
        question_file (str): Đường dẫn đến tệp JSON chứa thông tin encounter và image_id.
        definitions_file (str): Đường dẫn đến tệp JSON chứa thông tin qid, câu hỏi và câu trả lời.
        output_file (str): Đường dẫn để lưu tệp JSON kết quả.
    """
    final_vqa_data_list = []

    try:
        # 1. Đọc dữ liệu câu hỏi (chứa encounter_id, image_ids)
        print(f"Đang đọc dữ liệu câu hỏi từ: {question_file}")
        with open(question_file, 'r', encoding='utf-8') as f_ques:
            question_data = json.load(f_ques)
        print(f"Đã đọc {len(question_data)} mục từ tệp câu hỏi.")

        # 2. Đọc dữ liệu qid-câu trả lời
        print(f"Đang đọc dữ liệu QID-Answer từ: {definitions_file}")
        with open(definitions_file, 'r', encoding='utf-8') as f_qid:
            qid_answer_list = json.load(f_qid)
        print(f"Đã đọc {len(qid_answer_list)} mục từ tệp QID-Answer.")

        # 3. Lặp qua dữ liệu chính và kết hợp thông tin
        print("Đang kết hợp dữ liệu...")
        processed_count = 0
        skipped_count = 0

        # Lặp qua mỗi encounter
        for encounter_info in question_data:
            encounter_id = encounter_info.get('encounter_id')
            image_ids = encounter_info.get('image_ids')

            # Kiểm tra tính hợp lệ của dữ liệu encounter
            if not encounter_id:
                print(f"Cảnh báo: Bỏ qua encounter do thiếu 'encounter_id'. Dữ liệu: {encounter_info}")
                skipped_count += 1
                continue
            if not image_ids:
                print(f"Cảnh báo: Bỏ qua encounter do thiếu hoặc rỗng 'image_ids'. Dữ liệu: {encounter_info}")
                skipped_count += 1
                continue

            # Với mỗi encounter, lặp qua TẤT CẢ các cặp câu hỏi/trả lời
            for qa_info in qid_answer_list:
                qid = qa_info.get('qid')
                question = qa_info.get('question_en')
                answer = qa_info.get('options_en')

                # Kiểm tra tính hợp lệ của dữ liệu câu hỏi/trả lời
                if not qid:
                    print(f"Cảnh báo: Bỏ qua mục Q/A do thiếu 'qid'. Dữ liệu: {qa_info}")
                    continue
                if not question:
                    print(f"Cảnh báo: Bỏ qua mục Q/A do thiếu 'question_en'. Dữ liệu: {qa_info}")
                    continue
                if answer is None:
                    print(f"Cảnh báo: Bỏ qua mục Q/A do thiếu 'options_en'. Dữ liệu: {qa_info}")
                    continue

                # Tạo từ điển dữ liệu kết hợp cho cặp encounter-question này
                combined_data = {
                    "qid": qid,
                    "encounter_id": encounter_id,
                    "question": question,
                    "answer": answer,
                    "image_ids": image_ids,
                }

                final_vqa_data_list.append(combined_data)
                processed_count += 1

        # --- Ghi kết quả ---
        print(f"Đã xử lý và kết hợp thành công {processed_count} mục.")
        if skipped_count > 0:
            print(f"Đã bỏ qua {skipped_count} encounter do thiếu dữ liệu.")

        # 4. Ghi dữ liệu đã xử lý ra tệp JSON mới
        print(f"Đang ghi dữ liệu cuối cùng vào: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(final_vqa_data_list, f_out, indent=4, ensure_ascii=False)

        print(f"Tạo thành công tệp '{output_file}' với {len(final_vqa_data_list)} mục.")

    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp đầu vào. Chi tiết: {e}")
    except json.JSONDecodeError as e:
        print(f"Lỗi: Không thể giải mã JSON. Vui lòng kiểm tra định dạng tệp. Chi tiết: {e}")
    except KeyError as e:
        print(f"Lỗi: Thiếu khóa (key) mong đợi trong dữ liệu JSON. Khóa: {e}.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # --- Thiết lập argparse ---
    # 1. Tạo parser
    parser = argparse.ArgumentParser(
        description="Kết hợp dữ liệu câu hỏi VQA với các định nghĩa câu hỏi/trả lời thành một bộ dữ liệu duy nhất."
    )

    # 2. Thêm các tham số (arguments)
    parser.add_argument(
        '--question-file',
        type=str,
        required=True,
        help="Đường dẫn đến tệp JSON chứa encounter và image_ids (ví dụ: valid_ht_v2.json)."
    )
    parser.add_argument(
        '--definitions-file',
        type=str,
        required=True,
        help="Đường dẫn đến tệp JSON chứa qid, câu hỏi, và câu trả lời (ví dụ: closedquestions_definitions_imageclef2025.json)."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help="Đường dẫn cho tệp JSON đầu ra (ví dụ: val_vqa_dataset.json)."
    )
    
    # 3. Phân tích các tham số từ dòng lệnh
    args = parser.parse_args()

    # 4. Gọi hàm chính với các tham số đã được phân tích
    create_vqa_dataset(
        question_file=args.question_file,
        definitions_file=args.definitions_file,
        output_file=args.output_file
    )