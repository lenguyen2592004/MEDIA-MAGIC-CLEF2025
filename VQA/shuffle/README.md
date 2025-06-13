Create two shuffle version of vqa dataset for Gemini inferences.
```bash
python shuffle/shuffle.py --input val_gemini_vqa_dataset.json --output val_shuffle_1.json
python shuffle/shuffle.py --input val_gemini_vqa_dataset.json --output val_shuffle_2.json
```
Repeat this process for the test dataset, ensuring the output JSON files are named test_shuffle_1.json and test_shuffle_2.json respectively.
