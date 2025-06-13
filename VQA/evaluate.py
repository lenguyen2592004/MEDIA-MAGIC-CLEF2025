import sys
import json
import argparse # Thêm thư viện argparse

# Các hằng số định nghĩa QID và các nhóm QID cha.
# Những giá trị này không thay đổi, nên giữ nguyên.
QIDS = [
    "CQID010-001", "CQID011-001", "CQID011-002", "CQID011-003", "CQID011-004", 
    "CQID011-005", "CQID011-006", "CQID012-001", "CQID012-002", "CQID012-003", 
    "CQID012-004", "CQID012-005", "CQID012-006", "CQID015-001", "CQID020-001", 
    'CQID020-002', 'CQID020-003', 'CQID020-004', 'CQID020-005', 'CQID020-006', 
    'CQID020-007', 'CQID020-008', 'CQID020-009', "CQID025-001", "CQID034-001", 
    "CQID035-001", "CQID036-001",
]

QIDS_PARENTS = sorted(list(set([x.split('-')[0] for x in QIDS])))


def calculate_accuracy(qid2val_byencounterid_gold, qid2val_byencounterid_sys, qidparents=QIDS_PARENTS):
    """Tính toán điểm accuracy cho từng nhóm QID và tổng thể."""
    results = {}
    x_all, y_all = [], []
    encounter_ids = list(qid2val_byencounterid_gold.keys())

    for qid in qidparents:
        goldlist = [qid2val_byencounterid_gold[encounter_id][qid] for encounter_id in encounter_ids]
        syslist = [qid2val_byencounterid_sys[encounter_id][qid] for encounter_id in encounter_ids]
        x_all.extend(goldlist)
        y_all.extend(syslist)
        results[f'accuracy_{qid}'] = get_accuracy_score(goldlist, syslist)

    results['accuracy_all'] = get_accuracy_score(x_all, y_all)
    return results


def get_accuracy_score(gold_items, sys_items):
    """Tính điểm accuracy dựa trên sự giao nhau của các tập hợp câu trả lời."""
    total = 0
    weight_sum = 0
    for x, y in zip(gold_items, sys_items):
        weight = len(set(x).intersection(set(y)))
        weight_sum += weight / max(len(set(x)), len(set(y)))
        total += 1
    return weight_sum / total if total > 0 else 0


def organize_values(data):
    """Tổ chức lại dữ liệu từ danh sách các dict thành dict lồng nhau theo encounter_id và qid."""
    qid2val_byencounterid = {}
    for item in data:
        encounter_id = item['encounter_id'].split('-')[0]
        if encounter_id not in qid2val_byencounterid:
            qid2val_byencounterid[encounter_id] = {}
        
        for key, val in item.items():
            if key == 'encounter_id' or len(key) != 11:
                continue
            qid, _ = key.split('-')
            if qid not in qid2val_byencounterid[encounter_id]:
                qid2val_byencounterid[encounter_id][qid] = []
            qid2val_byencounterid[encounter_id][qid].append(val)
    return qid2val_byencounterid


def evaluate(reference_fn, prediction_fn, output_fn):
    """
    Hàm chính để thực hiện việc đánh giá.
    Đọc tệp tham chiếu và tệp dự đoán, tính toán điểm số và ghi kết quả ra tệp đầu ra.

    Args:
        reference_fn (str): Đường dẫn đến tệp JSON tham chiếu (ground-truth).
        prediction_fn (str): Đường dẫn đến tệp JSON dự đoán của hệ thống.
        output_fn (str): Đường dẫn để lưu tệp JSON chứa kết quả điểm số.
    """
    try:
        with open(reference_fn) as f:
            data_ref = json.load(f)
        with open(prediction_fn) as f:
            data_sys = json.load(f)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp - {e}", file=sys.stderr)
        sys.exit(1) # Thoát chương trình với mã lỗi
    except json.JSONDecodeError as e:
        print(f"Lỗi: Tệp JSON không hợp lệ - {e}", file=sys.stderr)
        sys.exit(1)

    print(f'Phát hiện {len(data_ref)} bản ghi trong tệp tham chiếu.', file=sys.stderr)
    print(f'Phát hiện {len(data_sys)} bản ghi trong tệp dự đoán.', file=sys.stderr)

    # Xử lý trường hợp tệp dự đoán rỗng
    if len(data_sys) == 0:
        results = {f"accuracy_{qid}": 0.0 for qid in QIDS_PARENTS}
        results["accuracy_all"] = 0.0
        results["number_cvqa_instances"] = len(data_ref)
    else:
        encounterids_ref = set([x['encounter_id'] for x in data_ref])
        encounterids_sys = set([x['encounter_id'] for x in data_sys])
        print(f'TRÙNG KHỚP ENCOUNTER_ID: {encounterids_ref == encounterids_sys}', file=sys.stderr)

        print('Đang tổ chức lại giá trị theo Question ID...', file=sys.stderr)
        qid2val_byencounterid_gold = organize_values(data_ref)
        qid2val_byencounterid_sys = organize_values(data_sys)

        print('Đang tính toán Accuracy...', file=sys.stderr)
        results = calculate_accuracy(qid2val_byencounterid_gold, qid2val_byencounterid_sys)
        results['number_cvqa_instances'] = len(encounterids_ref)
    
    # Ghi kết quả ra tệp
    print(f'Đang ghi kết quả vào {output_fn}...', file=sys.stderr)
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=4)
    
    print('Hoàn thành!', file=sys.stderr)


if __name__ == "__main__":
    # --- Thiết lập argparse ---
    parser = argparse.ArgumentParser(
        description="Đánh giá kết quả dự đoán CVQA bằng cách so sánh với tệp tham chiếu và tính toán điểm accuracy."
    )

    # Thêm các tham số dòng lệnh
    parser.add_argument(
        '--reference-file',
        type=str,
        required=True,
        help="Đường dẫn đến tệp JSON tham chiếu (ground-truth, ví dụ: valid_cvqa.json)."
    )
    parser.add_argument(
        '--prediction-file',
        type=str,
        required=True,
        help="Đường dẫn đến tệp JSON dự đoán của hệ thống (ví dụ: voted_cvqa_results.json)."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help="Đường dẫn để lưu tệp JSON chứa kết quả điểm số (ví dụ: scores_cvqa.json)."
    )

    # Phân tích các tham số
    args = parser.parse_args()

    # Gọi hàm xử lý chính với các tham số đã nhận được
    evaluate(
        reference_fn=args.reference_file,
        prediction_fn=args.prediction_file,
        output_fn=args.output_file
    )