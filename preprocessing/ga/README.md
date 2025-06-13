1. Install requirement libraries.
```bash
pip install -r ga/requirements.txt
```
2.Create enhanced images by genetic algorithm
```bash
python ga.py --input_dir images_final/images_train --output_dir enhance_train
python ga.py --input_dir images_final/images_valid --output_dir enhance_valid
python ga.py --input_dir images_final/images_test --output_dir enhance_test
```
