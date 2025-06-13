import json
import argparse 
from collections import Counter, defaultdict

def load_json_file(filepath):
    """Tải một file JSON và trả về nội dung của nó."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ - {filepath}")
        return None
    except Exception as e:
        print(f"Lỗi không xác định khi tải file {filepath}: {e}")
        return None

def vote_on_cvqa_data(file_paths):
    """
    Thực hiện vote câu trả lời cho các QID từ nhiều file JSON.

    Args:
        file_paths (list): Một danh sách các đường dẫn đến file JSON.

    Returns:
        list: Một danh sách các dictionary chứa dữ liệu đã được vote,
              hoặc một list rỗng nếu có lỗi.
    """
    all_data_sources = []
    for path in file_paths:
        data = load_json_file(path)
        if data:
            all_data_sources.append(data)
        else:
            print(f"Không thể tải dữ liệu từ file: {path}. Bỏ qua file này.")

    if not all_data_sources:
        print("Không có dữ liệu nào được tải. Dừng xử lý.")
        return []

    # Bước 1: Tổng hợp dữ liệu theo encounter_id và sau đó theo QID
    aggregated_data = defaultdict(lambda: defaultdict(list))
    all_encounter_ids = set()
    encounter_qid_order = {}

    if all_data_sources:
        first_data_source = all_data_sources[0]
        for record in first_data_source:
            enc_id = record.get("encounter_id")
            if enc_id and enc_id not in encounter_qid_order:
                encounter_qid_order[enc_id] = [k for k in record.keys() if k != "encounter_id"]

    for data_source_index, data_source in enumerate(all_data_sources):
        for encounter_record in data_source:
            encounter_id = encounter_record.get("encounter_id")
            if not encounter_id:
                print(f"Bỏ qua bản ghi không có 'encounter_id' trong file thứ {data_source_index + 1}")
                continue

            all_encounter_ids.add(encounter_id)
            if encounter_id not in encounter_qid_order:
                encounter_qid_order[encounter_id] = [k for k in encounter_record.keys() if k != "encounter_id"]

            for qid, answer in encounter_record.items():
                if qid == "encounter_id":
                    continue
                aggregated_data[encounter_id][qid].append(answer)

    sorted_encounter_ids = sorted(list(all_encounter_ids))

    # Bước 2: Thực hiện vote cho mỗi QID trong mỗi encounter
    voted_results = []
    for encounter_id in sorted_encounter_ids:
        voted_encounter = {"encounter_id": encounter_id}
        qids_for_this_encounter = encounter_qid_order.get(encounter_id, [])
        all_qids_for_encounter_aggregated = set(aggregated_data[encounter_id].keys())

        ordered_qids_to_process = []
        seen_qids_in_order = set()
        for qid in qids_for_this_encounter:
            if qid in all_qids_for_encounter_aggregated:
                ordered_qids_to_process.append(qid)
                seen_qids_in_order.add(qid)
        for qid in sorted(list(all_qids_for_encounter_aggregated - seen_qids_in_order)):
            ordered_qids_to_process.append(qid)

        for qid in ordered_qids_to_process:
            answers_list = aggregated_data[encounter_id].get(qid)
            if not answers_list:
                continue
            count = Counter(answers_list)
            most_common_answer = count.most_common(1)[0][0]
            voted_encounter[qid] = most_common_answer
        voted_results.append(voted_encounter)

    return voted_results

def save_json_file(data, filepath):
    """Lưu dữ liệu vào một file JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu thành công dữ liệu đã vote vào {filepath}")
    except IOError:
        print(f"Lỗi: Không thể ghi vào file - {filepath}")
    except Exception as e:
        print(f"Lỗi không xác định khi lưu file {filepath}: {e}")

def main(args):
    """
    Hàm chính để điều khiển luồng công việc: tải, vote, và lưu kết quả.
    """
    input_files = args.input
    output_file = args.output

    print(f"Bắt đầu quá trình vote cho các file: {', '.join(input_files)}")
    voted_data = vote_on_cvqa_data(input_files)

    if voted_data:
        save_json_file(voted_data, output_file)
        print("\n--- Hoàn tất! ---")
        print(f"Đã xử lý {len(voted_data)} bản ghi 'encounter'.")
    else:
        print("Không có dữ liệu nào được vote hoặc đã xảy ra lỗi trong quá trình xử lý.")

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Tạo parser
    parser = argparse.ArgumentParser(
        description="Thực hiện vote theo đa số cho câu trả lời từ nhiều file kết quả CVQA JSON.",
        epilog="Ví dụ sử dụng: python ten_script.py -i file1.json file2.json -o ket_qua.json"
    )

    # 2. Thêm các đối số
    parser.add_argument(
        "-i", "--input",
        nargs='+',  # Chấp nhận một hoặc nhiều giá trị
        required=True,
        help="Danh sách các đường dẫn đến các file JSON đầu vào (cách nhau bởi dấu cách)."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Đường dẫn đến file JSON đầu ra để lưu kết quả đã vote."
    )

    # 3. Phân tích cú pháp các đối số từ dòng lệnh
    args = parser.parse_args()

    # 4. Gọi hàm main với các đối số đã được phân tích
    main(args)