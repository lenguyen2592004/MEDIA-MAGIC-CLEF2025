## Requirements
Run the following command to install the required packages:
```bash
pip install -r MUMC/requirements.txt
```
## Usage
Please make sure Pretrain.yaml and VQA.yaml have correct directories.
1. Pretraining stage.
1.1. Training.
```bash
python MUMC/pretrain.py  \
--output_dir MUMC/output/pretrain \
--checkpoint MUMC/mumc_weights/pytorch/default/1/ALBEF.pth
```
2. Finetuning stage.
2.1. Training.
```bash
python train_vqa.py --dataset_use "clef2025" --output_dir "MUMC/output/finetune" --checkpoint MUMC/output/pretrain/latest_model.pth
```
2.2. Inference.
```bash
python MUMC/infer_one_sample.py \
  --checkpoint "MUMC/output/finetune/best_model.pth" \
  --device "cuda"
```
