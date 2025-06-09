Create vqa dataset
1. Create gemini vqa dataset
2. Create MUMC pretrain dataset
3. Create MUMC vqa dataset
```bash
python dataset/gemini_vqa_dataset.py --question-file "train_enrich_full.json" --definitions-file "closedquestions_definitions_imageclef2025.json"  --output-file "train_gemini_vqa_dataset.json"
python dataset/mumc_pretrain_data.py
python dataset/mumc_vqa_dataset.py
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named val_enrich_full.json and test_enrich_full.json respectively.
```bash
python dataset/mumc_pretrain_data.py --image_dir enhance_train --input_json train_enrich_full.json --output_json pretrain_data_train.json
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named pretrain_data_val.json and pretrain_data_test.json respectively.
```bash
python dataset/mumc_vqa_dataset.py --image_dir enhance_train --input_json train_enrich_full.json --output_json pretrain_data_train.json
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named pretrain_data_val.json and pretrain_data_test.json respectively.

CHUA XONGG
