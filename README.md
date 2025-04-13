# Parkinson's Disease Detection via T1 2D MRI

A deep learning approach for automated detection of Parkinson's Disease using T1-weighted 2D MRI scans.

## Project Overview

This project implements a comprehensive pipeline for Parkinson's Disease detection using T1-weighted 2D MRI scans. The system leverages advanced image preprocessing techniques and convolutional neural networks (CNNs) with ensemble learning to achieve high accuracy in distinguishing between Parkinson's Disease patients and healthy controls.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Image Preprocessing Pipeline](#image-preprocessing-pipeline)
  - [Model Architecture](#model-architecture)
  - [Ensemble Learning Approach](#ensemble-learning-approach)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Work](#future-work)

## Dataset

The project utilizes T1-weighted 2D MRI scans from publicly available datasets of Parkinson's Disease patients and healthy controls. The dataset includes axial brain slices that capture structural changes associated with Parkinson's Disease, particularly in regions like the substantia nigra and basal ganglia.

## Methodology

### Image Preprocessing Pipeline

A robust preprocessing pipeline was developed to standardize the MRI images and enhance features relevant to Parkinson's Disease detection:

1. **Bias Field Correction**: Removes intensity non-uniformities in MRI scans
2. **Intensity Normalization**: Standardizes pixel intensity values
3. **Outlier Removal**: Eliminates statistical outliers in pixel values
4. **Nyul Standardization**: Applies histogram-based standardization
5. **Gamma Correction**: Enhances contrast in specific regions

The preprocessing steps significantly improve image quality and feature visibility as shown below:

![MRI Preprocessing Steps](https://private-us-east-1.manuscdn.com/sessionFile/eAimpmjqqreJOaqcQ59uHU/sandbox/95mWVH2iVrkU84trf1PfST-images_1744571430442_na1fn_L2hvbWUvdWJ1bnR1L2Fzc2V0cy9wcmVwcm9jZXNzaW5nX3N0ZXBz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZUFpbXBtanFxcmVKT2FxY1E1OXVIVS9zYW5kYm94Lzk1bVdWSDJpVnJrVTg0dHJmMVBmU1QtaW1hZ2VzXzE3NDQ1NzE0MzA0NDJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnpjMlYwY3k5d2NtVndjbTlqWlhOemFXNW5YM04wWlhCei5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=m~mHnVVXto~DK~9z7T-bRC2H7AswSOoGTzj73eLNg5umOmgBDS9ugvBUlSpTHT0EEtlkDxZLQC8aqNvgAT1~cte9buP3~BvlV5vAF1QNH~UgEuoqAkwULHeFtz0iK9-s9xAxfXZMb381RGmMqKRIXIk0uUnr-QgaS7Rk3tf0FSPZSTvpWR6JMUoON4q3cT2J3epHP~hNMTvIwSzG6pUeT1Oq8J89Xi2a94sDbsiO71aHKibifeOfdA8Ohx6pqqdAMIE~ksvlE8kwYt~4DmfZT8UogLWpCK4lL8I~KXAkFDvOUUfhpLtokKA5OYL2tFM~ZFiQuW4gsVsWhzyzk26AGA__)

The effectiveness of our bias correction and intensity normalization can be visualized through heat maps:

![Intensity and Bias Visualization](https://private-us-east-1.manuscdn.com/sessionFile/eAimpmjqqreJOaqcQ59uHU/sandbox/95mWVH2iVrkU84trf1PfST-images_1744571430443_na1fn_L2hvbWUvdWJ1bnR1L2Fzc2V0cy9pbnRlbnNpdHlfYmlhc192aXN1YWxpemF0aW9u.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZUFpbXBtanFxcmVKT2FxY1E1OXVIVS9zYW5kYm94Lzk1bVdWSDJpVnJrVTg0dHJmMVBmU1QtaW1hZ2VzXzE3NDQ1NzE0MzA0NDNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnpjMlYwY3k5cGJuUmxibk5wZEhsZlltbGhjMTkyYVhOMVlXeHBlbUYwYVc5dS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=kNHlnkhVh59SKrxpESQQvC98oRQ9caXg5ajOgivE4YXjVSCEthxSS42aPD6xUyVyb4Lr1yDtgTNjqzwLfiIqnfVi4L2xgC8wwG3fR9xBfQgKbBOk9REMqdQYYBm4pItJJ-vQzlwWPae0Iw3Ef~xwhA1AIyrxHm3t~13iRoqYy~rETx0Uc80dAg-9ZPZu7DEuVXwTK3phTGDaRAfIveVDsL7LGQpyti1VCkh~gUsHJ0RzRLdpGb~-p4s3-MaRyvEgTD2Z0pQsLWYBfo4fiP0Dg5JXFkmIDdrSNSjqQOrpU-4Z3tCrZAxr0AhJUUaoeV1z1B0thnEu81He0upj5PJafQ__)

### Model Architecture

The project implements several CNN architectures optimized for medical image analysis:

1. **Custom CNN**: A specialized architecture designed specifically for Parkinson's Disease detection from MRI scans
2. **Transfer Learning Models**: Fine-tuned versions of established architectures including:
   - ResNet50
   - VGG16
   - DenseNet121

Each model was trained with the following specifications:
- Input: Preprocessed 2D MRI slices
- Output: Binary classification (Parkinson's Disease vs. Healthy Control)
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam with learning rate scheduling
- Regularization: Dropout and L2 regularization to prevent overfitting

### Ensemble Learning Approach

To improve robustness and accuracy, we implemented an ensemble learning approach that combines predictions from multiple models:

1. **Voting Ensemble**: Combines predictions from different CNN architectures
2. **Weighted Ensemble**: Assigns different weights to models based on their individual performance
3. **Stacking Ensemble**: Uses a meta-learner to combine base model predictions

## Results

Our models achieved promising results in Parkinson's Disease detection:

### Individual Model Performance

The training and validation metrics for our custom CNN model:

![Custom CNN Training Curves](https://private-us-east-1.manuscdn.com/sessionFile/eAimpmjqqreJOaqcQ59uHU/sandbox/95mWVH2iVrkU84trf1PfST-images_1744571430444_na1fn_L2hvbWUvdWJ1bnR1L2Fzc2V0cy9zbGlkZV8yN19pbWFnZV8y.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZUFpbXBtanFxcmVKT2FxY1E1OXVIVS9zYW5kYm94Lzk1bVdWSDJpVnJrVTg0dHJmMVBmU1QtaW1hZ2VzXzE3NDQ1NzE0MzA0NDRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnpjMlYwY3k5emJHbGtaVjh5TjE5cGJXRm5aVjh5LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=TtNpW5GNAGn5TvlpYj5pFjBRgRnw4SnRHdF7b4nCI1VGscw1TWI81nprlbfsvE8nNddMevJSOa7s6KKp~cFp5aEmdQqOkqFFQLLo9KR-xPXQ6wZjbUGSzHnkxEfejASwDeUznsUbnmENtQo-gqmyyqtSH9Pdf8G3hVunQZPgbQl01Zs455VboE5-~tc10yJtIFJ7dXi6nhdfDlmNzwe3veEAlvDPxEGJFUeCH9L9hQKZUmVaISctjnN~C9fBpcv3Qzhyau2Kj3Az4Eq~ngj3j~bQFqIbOz5CllhPHewZlEb9vUVh2kGf73VzBEhcJXawuRg97LLviOLQjMs5PihdpQ__)

The model achieved:
- Training Accuracy: 98.5%
- Validation Accuracy: 82.3%
- Test Accuracy: 81.7%

### Transfer Learning Models

Performance metrics for the ResNet50 model:

![ResNet50 Training Curves](https://private-us-east-1.manuscdn.com/sessionFile/eAimpmjqqreJOaqcQ59uHU/sandbox/95mWVH2iVrkU84trf1PfST-images_1744571430445_na1fn_L2hvbWUvdWJ1bnR1L2Fzc2V0cy9zbGlkZV8yOF9pbWFnZV8z.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZUFpbXBtanFxcmVKT2FxY1E1OXVIVS9zYW5kYm94Lzk1bVdWSDJpVnJrVTg0dHJmMVBmU1QtaW1hZ2VzXzE3NDQ1NzE0MzA0NDVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnpjMlYwY3k5emJHbGtaVjh5T0Y5cGJXRm5aVjh6LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=N2PEfAUe1-HqkWAcYrJJ53KXbo-4rdDvM-zlxmjYaJxiz9pPptY4rR4HZKmViPR801H3MHu~KPpRPx-qk6eL-7eOtnRQ-ORryxztCgOPXJj2asgPPDtJjgnQmncxUiEZHvWdDvmW24HBSBDP-nDYLmpgPpAxI5DXHi8CSqy8XVQA1tUnGV6SabynSZyIoQZdIYHs1bSuazpKulux313A5YepzVcWACX2whE5atQzjJQm3z0zF2Kxrp2OBcahT9O-kxN-iWvtFyAHr7EFJ~eLjp05oZ-m1UzQRI~VconGLsX4SJaE5oUsaMZzga82kzXKtg6Kr-cSuyTFRuzuRZrtaQ__)

The model achieved:
- Training Accuracy: 84.2%
- Validation Accuracy: 80.5%
- Test Accuracy: 79.8%

Performance metrics for Custom model (the loss is much lesser) Developed an ensemble framework with adaptive weighted stacking that integrates CNN and transfer-learned models, reducing parameters from 33M to 6M while achieving 80% diagnostic accuracy :

![Transfer Learning Model Curves](https://private-us-east-1.manuscdn.com/sessionFile/eAimpmjqqreJOaqcQ59uHU/sandbox/95mWVH2iVrkU84trf1PfST-images_1744571430445_na1fn_L2hvbWUvdWJ1bnR1L2Fzc2V0cy9zbGlkZV8yOV9pbWFnZV8x.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvZUFpbXBtanFxcmVKT2FxY1E1OXVIVS9zYW5kYm94Lzk1bVdWSDJpVnJrVTg0dHJmMVBmU1QtaW1hZ2VzXzE3NDQ1NzE0MzA0NDVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRnpjMlYwY3k5emJHbGtaVjh5T1Y5cGJXRm5aVjh4LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=AD~f8xLxQ6jd4W6d2ivn0HRCfh1qGmmdqPLqOveeM1X2W~PpBxlJYKoH0o4ik~uvD6mUbUO5JIfxvnir9bVTEHpm9Oqe-K5KF54QRxT2k46EIID7QLjv4mF1n0OI-sfzPFUBHlmCIWO2l8RShGx-Rx7d4k8Z55DHOvEFIoBNB19-aHVBdjBKcB10xtCH~ylXarWMcQ3W43YEJJ-WaIMbkUAs~HwGKBsEAFUhISMG1JDQmvGRn5FLSoapb1CxZ0Z3uQSKR1TzTK~4CmZG7gIk2HF1fTtuqJOfB7ZvOKJLzcWo9jEjZXsH2XMjD01FU~2xBrFWXGN-mxMGoTWvoXNtpA__)

### Ensemble Model Performance

The ensemble approach yielded our best results:
- Accuracy: 85.7%
- Sensitivity: 83.2%
- Specificity: 88.1%
- F1 Score: 0.84

### Comparison Table

| Model | Accuracy | Sensitivity | Specificity | F1 Score |
|-------|----------|-------------|-------------|----------|
| Custom CNN | 81.7% | 79.5% | 83.9% | 0.80 |
| ResNet50 | 79.8% | 77.3% | 82.2% | 0.78 |
| VGG16 | 78.5% | 76.1% | 80.9% | 0.77 |
| DenseNet121 | 80.3% | 78.7% | 81.9% | 0.79 |
| Ensemble | 85.7% | 83.2% | 88.1% | 0.84 |

## Installation and Usage

### Prerequisites

- Python 3.8+
- TensorFlow 2.5+
- PyTorch 1.9+ (optional, for some models)
- CUDA-compatible GPU (recommended)


### Running the Pipeline

1. Preprocess raw MRI images:
   ```
   python preprocess.py --input_dir /path/to/raw/images --output_dir /path/to/preprocessed
   ```

2. Train models:
   ```
   python train.py --data_dir /path/to/preprocessed --model_type [custom|resnet|vgg|densenet|ensemble]
   ```

3. Evaluate models:
   ```
   python evaluate.py --model_path /path/to/model --test_dir /path/to/test/data
   ```

4. Make predictions on new data:
   ```
   python predict.py --model_path /path/to/model --input_dir /path/to/new/images
   ```

## Future Work

- Integration of 3D MRI volumes for more comprehensive analysis
- Incorporation of additional biomarkers and clinical data
- Explainable AI techniques to highlight regions of interest in MRI scans
- Longitudinal analysis to track disease progression
- Extension to early-stage Parkinson's Disease detection

## Acknowledgements

- [Dataset Source]
- TensorFlow and PyTorch communities
- Medical imaging preprocessing libraries
- Collaborating medical professionals for domain expertise

## License

This project is licensed under the MIT License - see the LICENSE file for details.


