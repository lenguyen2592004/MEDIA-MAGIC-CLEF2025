# Qwen 2.5-VL for Dermatology VQA

This directory contains the implementation of Visual Question Answering for dermatology using the Qwen 2.5-VL model with voting mechanism and few-shot learning.

## Features

- **Model**: Qwen 2.5-VL 7B Instruct with 4-bit quantization
- **Voting Mechanism**: Multiple inference runs with answer shuffling for improved robustness
- **Few-shot Learning**: Uses training examples to improve performance on specific question types
- **GPU Optimization**: Memory-efficient implementation with automatic cleanup
- **Tiebreaker System**: Handles ties in voting with additional inference rounds

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Dataset Compatibility

This implementation requires the following dataset files that should be compatible with the existing MUMC and Gemini pipelines:

### Required Input Files:

1. **VQA Dataset Files** (created by [`dataset/gemini_vqa_dataset.py`](../dataset/gemini_vqa_dataset.py)):

   - `val_vqa_dataset.json` - Validation VQA dataset
   - `train_vqa_dataset.json` - Training VQA dataset

2. **Encounter Files** (enhanced image captions):

   - `valid_ht_v2.json` - Validation encounters with captions
   - `train_blip2_2b_6b_basic.json` - Training encounters with captions

3. **Answer Files**:

   - `train_cvqa.json` - Training answers (from original dataset)

4. **Image Directories**:
   - `images_final/images_final/images_valid/` - Validation images
   - `images_final/images_final/images_train/` - Training images
   - `images_final/images_final/images_test/` - Test images

### Dataset Format Compatibility:

The script expects the same dataset formats used by other components:

**VQA Dataset Format**:

```json
[
  {
    "qid": "CQID010-001",
    "encounter_id": "E001",
    "question": "What is the primary lesion type?",
    "answer": ["Option A", "Option B", "Option C", "Option D"],
    "image_ids": ["IMG_001.jpg", "IMG_002.jpg"]
  }
]
```

**Encounter Format**:

```json
[
  {
    "encounter_id": "E001",
    "image_ids": ["IMG_001.jpg", "IMG_002.jpg"],
    "query_content_en": "Enhanced caption describing the dermatological condition..."
  }
]
```

**CVQA Answer Format**:

```json
[
  {
    "encounter_id": "E001",
    "CQID010-001": 2,
    "CQID011-001": 0,
    ...
  }
]
```

## Usage

### Basic Inference

Run inference on validation set:

```bash
python qwen2_vl_inference.py \
  --val_vqa_dataset path/to/val_vqa_dataset.json \
  --valid_ht_v2 path/to/val_enrich_full.json \
  --train_vqa_dataset path/to/train_vqa_dataset.json \
  --train_json path/to/train_blip2_2b_6b_basic.json \
  --train_cvqa path/to/train_cvqa.json \
  --image_dir path/to/images_final/images_final \
  --task valid \
  --output_file qwen_prediction.json \
  --num_gpus 2
```

### Test Set Inference

For test set inference:

```bash
python qwen2_vl_inference.py \
  --val_vqa_dataset path/to/test_vqa_dataset.json \
  --valid_ht_v2 path/to/test_enrich_full.json \
  --train_vqa_dataset path/to/train_vqa_dataset.json \
  --train_json path/to/train_blip2_2b_6b_basic.json \
  --train_cvqa path/to/train_cvqa.json \
  --image_dir path/to/images_final/images_final \
  --task test \
  --output_file qwen_test_prediction.json \
  --num_gpus 2
```

### Arguments

- `--val_vqa_dataset`: Path to the main VQA dataset file (validation or test)
- `--valid_ht_v2`: Path to the encounter file with image captions
- `--train_vqa_dataset`: Path to training VQA dataset for few-shot examples
- `--train_json`: Path to training encounters with captions
- `--train_cvqa`: Path to training answers file
- `--image_dir`: Root directory containing image folders
- `--task`: Task type (`valid`, `test`, `train`)
- `--output_file`: Output JSON file path
- `--num_gpus`: Number of GPUs to use (default: 2)

## Model Configuration

The implementation uses:

- **Model**: `Qwen/Qwen2-VL-7B-Instruct`
- **Quantization**: 4-bit with NF4 quantization type
- **Image Processing**: 480x640 resolution with standardized normalization
- **Memory Management**: Automatic GPU cache clearing between inferences

## Output Format

The output follows the same format as other VQA components for easy integration with the ensemble system:

```json
[
  {
    "encounter_id": "E001",
    "CQID010-001": 2,
    "CQID011-001": 0,
    "CQID012-001": 1,
    ...
  }
]
```

## Integration with Ensemble

The output can be directly used with the ensemble voting system in [`../ensemble/`](../ensemble/) along with outputs from MUMC and Gemini models.

## Performance Features

- **Voting Mechanism**: Reduces variance by running multiple inferences with shuffled answer orders
- **Few-shot Learning**: Automatically selects relevant training examples for each question type
- **Memory Optimization**: Efficient GPU memory management for large models
- **Partial Results**: Saves intermediate results during processing for recovery

## Hardware Requirements

- **GPU Memory**: Minimum 12GB per GPU (RTX 3060 12GB or better)
- **System RAM**: 16GB+ recommended
- **Storage**: ~20GB for model weights and datasets

## Troubleshooting

1. **Out of Memory Error**: Reduce `num_gpus` or use smaller batch sizes
2. **Import Errors**: Ensure all dependencies are installed with correct versions
3. **Dataset Format Issues**: Verify that input files match the expected JSON schemas
4. **Model Loading Issues**: Check internet connection for initial model download
