# Ensembled Model

This folder contains a Python script eval2.py that performs out-of-distribution (OOD) detection evaluation for image classification models. The folder also has a configs sub-directory containing subdirectories knn, models, and util which contain utility functions and configurations for the evaluation. There is also a score2.py module that provides functions for computing scores for OOD detection.

Here is a breakdown of the code:

`eval2.py`: This is the main Python script that performs the OOD detection evaluation. It loads command-line arguments using argparse, and uses several utility functions and modules to load data, model, and compute metrics.

`configs/`: This is a directory containing subdirectories knn, models, and util which contain utility functions and configurations for the evaluation.

`score2.py`: This is a module that provides functions for computing scores for OOD detection.
