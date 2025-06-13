Installing requirement libraries.
```bash
pip install -r requirements.txt
```
Using linking external knowledge to enrich medical caption
```bash
python linking.py \
    --input merge_train.json \
    --output train_enrich_full.json \
    --model gemini-2.5-flash-preview-04-17 \
    --api-keys KEY_1_HERE KEY_2_HERE KEY_3_HERE \
    --requests-per-key 100 \
    --checkpoint-freq 50
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named val_enrich_full.json and test_enrich_full.json respectively.
