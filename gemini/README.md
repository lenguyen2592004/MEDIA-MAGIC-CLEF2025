Installing requirement libraries.
```bash
pip install -r requirements.txt
```
Using Gemini to generate answer. We will generate three inferences for three version datasets.
```bash
python gemini_infer.py \
    --vqa_file test_gemini_vqa_dataset.json \
    --ht_file test_enrich_full.json \
    --image_dir enhance_test \
    --output_file gemini_1_test.json \
    --model_name gemini-2.5-flash-preview-04-17 \
    --api_key YOUR_API_KEY
```
Repeat this process for other test datasets, ensuring the output JSON files are named gemini_2_test.json and gemini_3_test.json respectively.
