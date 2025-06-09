Create two shuffle version of vqa dataset for Gemini inferences.
```bash
python shuffle/shuffle.py --input val_gemini_vqa_dataset.json --output shuffled_output.json
python shuffle/shuffle.py --input test_gemini_vqa_dataset.json --output shuffled_output.json
```
