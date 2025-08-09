# H3N1 at MEDIQA-MAGIC 2025: DermoSegDiff and DermKEM for Comprehensive Dermatology AI

This project includes two tasks: **Skin Lesion Segmentation** and **Closed Visual Question Answering (cVQA)** for dermatological images. 

The segmentation task is a modified version of **DermoSegDiff**, adapted for a new dataset. The VQA task develop **DermKEM** system that enables answering questions about skin lesion images. Each task is implemented in a separate directory with its own detailed README.

Tasks

- Segmentation: Based on DermoSegDiff (MICCAI 2023 PRIME Workshop), using Denoising Diffusion Probabilistic Models for skin lesion segmentation. See [SEGMENTATION/README](./SEGMENTATION/README.md) for details.

- VQA: Visual Question Answering by DermKEM system for skin lesion images. See [VQA/README](./VQA/README.md) for details.
- For both tasks, we use the same genetic algorithm, see [ga](./ga) for details.
