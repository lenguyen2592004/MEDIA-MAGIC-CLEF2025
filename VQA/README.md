# DermKEM (Dermatology Knowledge-Enhanced Ensemble Model) system for Dermatology VQA
This repository contains the official implementation for our paper "Hoangwithhisfriends at MEDIQA-MAGIC 2025:
DermoSegDiff and DermKEM for Comprehensive Dermatology AI" in task 2: Visual Question Answering for Dermatology VQA.
# Structure
Please ensure download full dataset and unzip. The structure should be:

project-root/

├── 📁 MUMC/

├── 📁 dataset/

├── 📁 ensemble/

├── 📁 gemini/

├── 📁 preprocessing/

├── 📁 shuffle/

├── 📁 images_final/
    ├── 📁 images_train/
    ├── 📁 images_valid/
    ├── 📁 images_test/


├── 📄 evaluate.py

└── 📄 README.md

# Install
## Python environment
```bash
python>=3.13.4
```
## Usage

To use the components of this project, please follow the instructions in the `README.md` file located within each corresponding directory.

### 1. Preprocessing

- **1.1. Image Enhancement:**
  - Follow the guide in the [`ga`](../ga) directory.

- **1.2. Additional Caption Generation:**
  - Follow the guide in the [`preprocessing/blip`](./preprocessing/blip) directory.

- **1.3. Concatenate Captions:**
  - Follow the guide in the [`preprocessing/concat_caption`](./preprocessing/concat_caption) directory.

- **1.4. Linking External Knowledge:**
  - Follow the guide in the [`preprocessing/linking_external_knowledge`](./preprocessing/linking_external_knowledge) directory.

### 2. Creating the Dataset

- Follow the guide in the [`dataset`](./dataset) directory.

### 3. Creating the Shuffled Dataset

- Follow the guide in the [`shuffle`](./shuffle) directory.

### 4. Baseline Models

- **4.1. MUMC:**
  - Follow the guide in the [`MUMC`](./MUMC) directory.

- **4.2. Gemini 2.5:**
  - Follow the guide in the [`gemini`](./gemini) directory. 

### 5. Ensemble

- Follow the guide in the [`ensemble`](./ensemble) directory.
