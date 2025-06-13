Merge base caption with blip generated caption
```bash
python merge_script.py train.json train_blip -o merge_train.json
```
Repeat this process for the valid and test datasets, ensuring the output JSON files are named merge_val.json and merge_test.json respectively.
