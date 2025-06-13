# DermoSegDiff: A Boundary-aware Segmentation Diffusion Model for Skin Lesion Delineation <br> <span style="float: right"><sub><sup>MICCAI 2023 PRIME Workshop</sub></sup></span>
[![arXiv](https://img.shields.io/badge/arXiv-2309.00108-b31b1b.svg)](https://arxiv.org/abs/2308.02959)

This project is a modified version of the original [DermoSegDiff](https://github.com/xmindflow/DermoSegDiff) repository, adapted to work with a new dataset from competition [ImageCLEFmed MEDIQA-MAGIC 2025](https://www.imageclef.org/2025/medical/mediqa) and updated training/testing configurations. The original work, presented at the MICCAI 2023 PRIME Workshop, proposes a novel framework for skin lesion segmentation using Denoising Diffusion Probabilistic Models (DDPMs) with a boundary-aware loss function and a U-Net-based denoising network. Our modifications include support for a new dataset, updated training/testing scripts, and the ability to continue training from checkpoints.

For details on the original methodology, refer to the original repository and paper cited below.

## Network
<p align="center">
  <em>Network</em><br/>
  <img width="600" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/7619985e-d894-4ada-9125-9f40a32bae7d">
  <br/>
  <br/>
  <em>Method</em></br>
  <img width="800" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/61879630/0919e613-972a-47ac-ac79-04a2ae51ed1e">
</p>

## Citation
```bibtex
@inproceedings{bozorgpour2023dermosegdiff,
  title={DermoSegDiff: A Boundary-Aware Segmentation Diffusion Model for Skin Lesion Delineation},
  author={Bozorgpour, Afshin and Sadegheih, Yousef and Kazerouni, Amirhossein and Azad, Reza and Merhof, Dorit},
  booktitle={Predictive Intelligence in Medicine},
  pages={146--158},
  year={2023},
  organization={Springer Nature Switzerland}
}
```
<p align="center">
  <img width="620" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/30bb1483-e9f8-44df-bede-13238df6f4f0">
</p>

## Acknowledgments
This project is built upon the original DermoSegDiff repository. We thank the authors for their foundational work. Modifications include support for a new dataset and updated training/testing pipelines.

## News
- July 25, 2023: Accepted in MICCAI 2023 PRIME Workshop! 🥳
- June 2025: Adapted DermoSegDiff for a new dataset with updated training and testing scripts.



### Requirements

- Ubuntu 16.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.7 or higher
- Hardware Spec
- GPU with 12GB memory or larger capacity (With low GPU memory you need to change and decrease `dim_x_mults`, `dim_g_mults`, `dim_x`, and `dim_g` params. You also need to change `batch_size` respectively. If you tune it well you won't lose considerable capability!)
- _For our experiments, we used 1GPU P100 (kaggle)_


Install dependencies using:

`pip install -r requirements.txt`

### Model weights
The original pretrained weights are available from the original repository:
Dataset   | Model          | download link 
-----------|----------------|----------------
ISIC2018  | DermoSegDiff-A | [[Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EhsfBqr1Z-lCr6KaOkRM3EgBIVTv8ew2rEvMWpFFOPOi1w?e=ifo9jF)] 
PH2       | DermoSegDiff-B | [[Download](https://uniregensburg-my.sharepoint.com/:f:/g/personal/say26747_ads_uni-regensburg_de/EoCkyNc5yeRFtD-KTFbF0gcB8lbjMLY6t1D7tMYq7yTkfw?e=tfGHee)] 
For the new dataset, you can use the provided fine-tuned checkpoint (e.g., [n-dsd_h01_s-128_b-8_t-250_sc-linear_best.pth](https://www.kaggle.com/code/sscarecrow/finetune-dermosegdiff-part-9)) or train your own model.

## How to use
  ### Training `src`
  To train the model on the new dataset, use the provided configuration file and run the following command from the project root:
  
  ```!python src/training.py --config configs/new_dataset/dermosegdiff/dsd_01.yaml```
  
  To continue training from a checkpoint, modify the following fields in `configs/new_dataset/dermosegdiff/dsd_01.yaml`:
- continue_training: Set to true to resume training.
- auto_continue: Set to true to automatically load the latest checkpoint.
- save_checkpoint_dir: Directory to save new checkpoints.
- load_dir: Directory containing the checkpoint to load.
- save_dir: Directory to save model outputs.
  You can also adjust hyperparameters (e.g., batch_size, learning_rate, timesteps, etc.) in the same YAML file.
  
  ### Testing
  
  To evaluate the model on the test set, use the following command, specifying the config file and the path to the best model checkpoint:
  
```
  !python src/testing.py \
  -c configs/new_dataset/dermosegdiff/dsd_01.yaml \
  --best_model_path "/kaggle/input/finetune-dermosegdiff-part-9/checkpoints/n-dsd_h01_s-128_b-8_t-250_sc-linear_best.pth" 
```
  
  ### Testing on a Single Sample
``` 
    !python src/testing_one_sample.py \
    -c configs/new_dataset/dermosegdiff/dsd_01.yaml \
    --image_path "/kaggle/input/imageclefmed-mediqa-magic-2025/images_final/images_final/images_valid/IMG_ENC00853_00001.jpg" \
    --ensemble_number 5 \
    --best_model_path "/kaggle/input/finetune-dermosegdiff-part-9/checkpoints/n-dsd_h01_s-128_b-8_t-250_sc-linear_best.pth"
```
  
  ### Configuration
  The configuration file (dsd_01.yaml) contains all necessary parameters for training and testing. Key fields include:
- Dataset paths
- Model architecture settings (dim_x_mults, dim_g_mults, etc.)
- Training hyperparameters (batch_size, learning_rate, timesteps, etc.)
- Checkpoint loading/saving options

Refer to the YAML file for detailed descriptions of each parameter.
  ### Evaluation
  
  <p align="center">
    <img width="800" alt="image" src="https://github.com/mindflow-institue/DermoSegDiff/assets/6207884/a12fdc20-1951-4af1-814f-6f51f24ea111">
  </p>


## References
- Original DermoSegDiff Repository: https://github.com/xmindflow/DermoSegDiff
- Denoising Diffusion Implementation: https://github.com/lucidrains/denoising-diffusion-pytorch

