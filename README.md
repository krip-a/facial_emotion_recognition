# Facial Emotion Recognition using CNN
## Overview
This project explores facial emotion recognition using Convolutional Neural Networks (CNNs) and deep learning.

The model classifies facial images into seven emotion categories:

- Angry

- Disgust

- Fear

- Happy

- Neutral

- Sad

- Surprise

This project uses the FER-2013 dataset and compares CNN architechtures and training strategies to evaluate model performance and generalization. Through iterative experimentation, the model's test accuracy improved from 45.11% to 52.93% by introducing regularization and improving the training and validation process.

## Dataset
Download the FER-2013 Dataset from [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
and place it in the data/ directory

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- COnvolutional Neural Network (CNN)

Through iterative experimentation, the model's test accuracy improved from 45.11% to 52.93% by introducing regularization and improving the training and validation process.

## Methodology

1. Data Loading and Preprocessing

    - Loaded facial images from emotion-specific directories.

    - Converted images to grayscale.

    - Normalized pixel values to the range [0, 1].

    - Encoded emotion labels into numerical categories.

    - Reshaped images to include a channel dimension for CNN input.

2. Model Development

    - Developed a CNN architecture using:

    - Convolutional layers for image feature extraction.

    - Pooling layers for spatial downsampling.

    - Dense layers for classification.

    - A seven-neuron output layer with softmax activation for multiclass predictions.

3. Training and Evaluation

    - Trained and evaluated multiple CNN configurations to investigate overfitting and model generalization.

    - Training improvements included:

    - Dropout regularization.

    - Early stopping based on validation loss.

    - Stratified training-validation splitting.


Model performance was evaluated using validation accuracy, test accuracy, and test loss.

## Record of Experiments

|#| Model  |      Training Changes     | Validation Accuracy | Test Accuracy |
|-|--------|---------------------------|---------------------|---------------|
|1| CNN v1 | Basic CNN, no Dropout     |       11.18%        |     45.11%    |
|2| CNN v2 |Dropout+EarlyStopping+stratified split| 51.46%   |     52.93%    |

### Experiment Notes

CNN v1 was used as a baseline model. The model achieved high training accuracy but poor validation performance, indicating severe overfitting.

CNN v2 improved generalization by introducing dropout regularization, early stopping, and a stratified train-validation split. The validation and test accuracies became more consistent, indicating a more reliable evaluation pipeline.

## Results
CNN v2 achieved the best test accuracy of 52.93%, compared with 45.11% for the baseline model. This is an improvement of 7.82 percentage points.

Although performance improved, facial emotion recognition remains challenging due to differences in facial expressions, image quality, and similarities between emotion categories.

## Future Improvements

1. Apply data augmentation to improve generalization.

2. Experiment with deeper CNN architectures.

3. Address class imbalance using class weights.

4. Evaluate model performance using confusion matrices and per-class metrics.

5. Develop an interactive application for predicting emotions from uploaded images.