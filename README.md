# DermKEM (Dermatology Knowledge-Enhanced Ensemble Model) system for Dermatology VQA
This repository contains the official implementation for our paper "Hoangwithhisfriends at MEDIQA-MAGIC 2025:
DermoSegDiff and DermKEM for Comprehensive Dermatology AI" in task 2: Visual Question Answering for Dermatology VQA.
# Install
# # Python environment
```bash
python>=3.13.4
```
## Usage
### 1. Create dataset
```bash
python dataset/gemini_vqa_dataset.py --question-file "path/to/your/questions.json" --definitions-file "path/to/your/closedquestions_definitions_imageclef2025.json"  --output-file "path/to/your/output.json"
python dataset/mumc_pretrain_data.py
python dataset/mumc_vqa_dataset.py
```
### 2. Preprocessing
#### 2.1. GA images.
```bash
python ga.py --input_dir <path_to_your_input_image_folder> --output_dir <path_to_your_output_image_folder>
```
