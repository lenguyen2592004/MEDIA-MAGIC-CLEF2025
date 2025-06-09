# DermKEM (Dermatology Knowledge-Enhanced Ensemble Model) system for Dermatology VQA
This repository contains the official implementation for our paper "Hoangwithhisfriends at MEDIQA-MAGIC 2025:
DermoSegDiff and DermKEM for Comprehensive Dermatology AI" in task 2: Visual Question Answering for Dermatology VQA.
# Install
# # Python environment
```bash
python>=3.13.4
```
## Usage

Để sử dụng các thành phần của dự án, vui lòng tham khảo hướng dẫn chi tiết trong file `README.md` tại mỗi thư mục tương ứng.

### 1. Preprocessing (Tiền xử lý)

- **1.1. Image Enhancement (Cải thiện ảnh):**
  - Xem hướng dẫn trong thư mục [`preprocessing/ga`](./preprocessing/ga).

- **1.2. Additional Caption Generation (Tạo chú thích bổ sung):**
  - Xem hướng dẫn trong thư mục [`preprocessing/blip`](./preprocessing/blip).

- **1.3. Concatenate Caption (Kết hợp chú thích):**
  - Xem hướng dẫn trong thư mục [`preprocessing/concat_caption`](./preprocessing/concat_caption).

- **1.4. Linking External Knowledge (Liên kết tri thức ngoài):**
  - Xem hướng dẫn trong thư mục [`preprocessing/linking_external_knowledge`](./preprocessing/linking_external_knowledge).

### 2. Creating Dataset (Tạo bộ dữ liệu)

- Xem hướng dẫn trong thư mục [`dataset`](./dataset).

### 3. Creating Shuffling Dataset (Tạo bộ dữ liệu đã xáo trộn)

- Xem hướng dẫn trong thư mục [`shuffle`](./shuffle).

### 4. Baseline Models (Các mô hình cơ sở)

- **4.1. MUMC:**
  - Xem hướng dẫn trong thư mục [`MUMC`](./MUMC).

- **4.2. Gemini 2.5:**
  - Xem hướng dẫn trong thư mục [`Gemini-2.5`](./Gemini-2.5). 
  *(Lưu ý: Tôi đã sửa đường dẫn này từ `MUMC` thành `Gemini-2.5` để khớp với tiêu đề. Bạn hãy chỉnh lại cho đúng với tên thư mục thực tế của bạn).*

### 5. Ensemble (Tổ hợp mô hình)

- Xem hướng dẫn trong thư mục [`ensemble`](./ensemble).
