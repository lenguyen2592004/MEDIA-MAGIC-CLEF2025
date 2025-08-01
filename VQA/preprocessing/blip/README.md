Installing requirement libraries.

```bash
pip install -r requirements.txt
```

Please download SkinCAP dataset [`Link`](https://huggingface.co/datasets/joshuachou/SkinCAP) and put it in [`./dermatology-dataset`](./dermatology-dataset) directory.

Generating additional caption for images via BLIP caption model

1. Finetuning on SkinCAP dataset

```bash
python finetune_blip.py \
    --json_file "dermatology-dataset/skincap_v240623.json" \
    --image_dir "dermatology-dataset/skincap" \
    --output_dir "./skincap_blip_finetuned" \
    --num_epochs 8 \
    --batch_size 8 \
    --learning_rate 5e-5
```

2. Infernce on DermaVQA-DAS dataset

```bash
python inference_blip.py \
    --input-json train.json \
    --image-dir enhance_train \
    --output-json train_blip.json \
    --model-name "./skincap_blip_finetuned" \
    --device "cpu"
```

Repeat this process for the valid and test datasets, ensuring the output JSON files are named val_blip.json and test_blip.json respectively.
