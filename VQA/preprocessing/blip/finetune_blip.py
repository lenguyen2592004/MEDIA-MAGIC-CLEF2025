%%writefile finetune_blip.py
import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import shutil
import argparse

# --- Helper Functions ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune mô hình BLIP để tạo chú thích hình ảnh y tế.")

    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help="Đường dẫn đến file JSON chứa chú thích và tên file ảnh."
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="Đường dẫn đến thư mục chứa file ảnh."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Salesforce/blip-image-captioning-base',
        help="Tên hoặc đường dẫn của mô hình pre-trained từ Hugging Face Hub."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./skincap_blip_finetuned',
        help="Thư mục để lưu mô hình đã fine-tune và các checkpoint."
    )
    parser.add_argument('--num_epochs', type=int, default=40, help="Số lượng epochs để huấn luyện.")
    parser.add_argument('--batch_size', type=int, default=16, help="Kích thước batch size.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Tốc độ học (learning rate).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed để tái tạo kết quả.")
    
    if 'ipykernel' in __import__('sys').modules:
        cmd_args = [
            '--json_file', '/kaggle/input/dermatology-dataset/skincap_v240623.json',
            '--image_dir', '/kaggle/input/dermatology-dataset/skincap',
            '--output_dir', './skincap_blip_finetuned_v2',
            '--num_epochs', '10', # Giảm epoch để chạy thử nhanh hơn
            '--batch_size', '8'
        ]
        return parser.parse_args(cmd_args)
    else:
        return parser.parse_args()



TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MAX_LENGTH = 128
EARLY_STOPPING_PATIENCE = 3

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# --- Dataset Class ---
class SkincapDataset(Dataset):
    def __init__(self, data_list, image_dir, processor, max_length, include_raw_caption=False):
        self.data_list = data_list
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.include_raw_caption = include_raw_caption

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_filename = item['image']
        caption = item['caption']

        image_path = os.path.join(self.image_dir, image_filename)
        try:
            raw_image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return None
        except Exception as e:
            return None

        encoding = self.processor(
            images=raw_image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        processed_item = {k: v.squeeze() for k, v in encoding.items()}
        if self.include_raw_caption:
            processed_item['raw_caption'] = caption
        return processed_item

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = input_ids.clone()
    collated_batch = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    if 'raw_caption' in batch[0]:
        collated_batch['raw_caption'] = [item['raw_caption'] for item in batch]
    return collated_batch

if __name__ == '__main__':
    args = parse_arguments()

    JSON_FILE = args.json_file
    IMAGE_DIR = args.image_dir
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    BEST_MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'best_model_checkpoint')

    RANDOM_SEED = args.seed
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(BEST_MODEL_CHECKPOINT_DIR)

    print(f"Loading data from {JSON_FILE}...")
    with open(JSON_FILE, 'r') as f:
        all_data = json.load(f)

    random.shuffle(all_data)
    n_total = len(all_data)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    train_data = all_data[:n_train]
    val_data = all_data[n_train : n_train + n_val]
    test_data = all_data[n_train + n_val :]

    print(f"Total samples: {n_total}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    print(f"Loading model and processor: {MODEL_NAME}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    train_dataset = SkincapDataset(train_data, IMAGE_DIR, processor, MAX_LENGTH)
    val_dataset = SkincapDataset(val_data, IMAGE_DIR, processor, MAX_LENGTH)
    test_dataset = SkincapDataset(test_data, IMAGE_DIR, processor, MAX_LENGTH, include_raw_caption=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_completed_epochs = 0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        training_completed_epochs = epoch + 1
        model.train()
        train_loss_total = 0
        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        for batch_idx, batch in enumerate(progress_bar_train):
            if batch is None: continue
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_total += loss.item()
                progress_bar_train.set_postfix({'loss': loss.item()})
            else:
                print(f"Warning: Loss is None for training batch {batch_idx}. Skipping backward pass.")
        avg_train_loss = train_loss_total / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss_total = 0
        progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar_val):
                if batch is None: continue
                batch_input = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch_input)
                loss = outputs.loss
                if loss is not None:
                    val_loss_total += loss.item()
                    progress_bar_val.set_postfix({'val_loss': loss.item()})
        avg_val_loss = val_loss_total / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model checkpoint...")
            model.save_pretrained(BEST_MODEL_CHECKPOINT_DIR)
            processor.save_pretrained(BEST_MODEL_CHECKPOINT_DIR)
        else:
            epochs_no_improve += 1
            print(f"Validation loss ({avg_val_loss:.4f}) did not improve for {epochs_no_improve} epoch(s) from best ({best_val_loss:.4f}).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
            break
    if training_completed_epochs < NUM_EPOCHS and epochs_no_improve < EARLY_STOPPING_PATIENCE :
         print(f"Training finished after {training_completed_epochs} epochs (completed all planned epochs).")
    elif epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Training stopped early after {training_completed_epochs} epochs.")
    else:
        print(f"Training completed all {NUM_EPOCHS} epochs.")


    print(f"\nLoading best model from {BEST_MODEL_CHECKPOINT_DIR} for final operations...")
    try:
        model = BlipForConditionalGeneration.from_pretrained(BEST_MODEL_CHECKPOINT_DIR)
        processor = BlipProcessor.from_pretrained(BEST_MODEL_CHECKPOINT_DIR)
        model.to(device)
        print("Successfully loaded best model checkpoint.")
    except OSError:
        print(f"Warning: Could not load best model checkpoint from {BEST_MODEL_CHECKPOINT_DIR}. Using the model from the last epoch.")


    print(f"Saving final (best) model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning and saving complete.")

    print("\n--- Inference on Test Set and BLEU Score Calculation ---")
    model.eval()
    all_reference_captions = []
    all_generated_captions = []
    chencherry = SmoothingFunction()

    with torch.no_grad():
        progress_bar_test = tqdm(test_dataloader, desc="Testing")
        for batch_idx, batch in enumerate(progress_bar_test):
            if batch is None: continue
            pixel_values = batch['pixel_values'].to(device)
            reference_captions_batch = batch.get('raw_caption', [])
            if not reference_captions_batch: continue

            generated_ids = model.generate(pixel_values=pixel_values, max_length=MAX_LENGTH, num_beams=4, early_stopping=True)
            generated_captions_batch = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_reference_captions.extend(reference_captions_batch)
            all_generated_captions.extend(generated_captions_batch)

    bleu_scores = []
    if len(all_reference_captions) == len(all_generated_captions) and len(all_reference_captions) > 0:
        print("\nCalculating BLEU scores...")
        for ref_text, gen_text in zip(all_reference_captions, all_generated_captions):
            ref_tokens = [word_tokenize(ref_text.lower())]
            gen_tokens = word_tokenize(gen_text.lower())
            try:
                score = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
                bleu_scores.append(score)
            except Exception:
                bleu_scores.append(0.0)
        if bleu_scores:
            avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
            print(f"Average BLEU-4 Score on Test Set: {avg_bleu_score:.4f}")
            # print("\n--- Sample Test Set Inferences ---")
            # num_samples_to_print = min(5, len(all_reference_captions))
            # for i in range(num_samples_to_print):
            #     print(f"Sample {i+1}:")
            #     print(f"  True Caption: {all_reference_captions[i]}")
            #     print(f"  Generated Caption: {all_generated_captions[i]}")
            #     print(f"  BLEU-4 Score: {bleu_scores[i]:.4f}\n")
        else:
            print("No BLEU scores to calculate (list was empty).")
    else:
        print("Mismatch in number of reference and generated captions, or no test data processed for BLEU.")