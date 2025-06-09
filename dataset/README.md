Create vqa dataset
1. Create gemini vqa dataset
```bash
python dataset/gemini_vqa_dataset.py --question-file "train_enrich_full.json" --definitions-file "closedquestions_definitions_imageclef2025.json"  --output-file "train_gemini_vqa_dataset.json"
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named val_gemini_vqa_dataset.json and test_gemini_vqa_dataset.json respectively.
2. Create MUMC pretrain dataset
```bash
python dataset/mumc_pretrain_data.py --image_dir enhance_train --input_json train_enrich_full.json --output_json pretrain_data_train.json
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named pretrain_data_val.json and pretrain_data_test.json respectively.
3. Create MUMC vqa dataset
```bash
python dataset/mumc_vqa_dataset.py \
    --train_encounters_json train_enrich_full.json \
    --train_answers_json train_cvqa.json \
    --valid_encounters_json val_enrich_full.json \
    --valid_answers_json valid_cvqa.json \
    --test_encounters_json val_enrich_full.json \
    --questions_json closedquestions_definitions_imageclef2025.json \
    --train_image_dir enhance_train \
    --valid_image_dir enhance_valid \
    --test_image_dir enhance_test \
    --output_dir mumc_vqa_dataset
```
