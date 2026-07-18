# Facial Emotion Recognition using CNN


## Dataset
Download the FER-2013 Dataset from [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
and place it in the data/ directory

## Record of Experiments

|#| Model  |      Training Changes     | Validation Accuracy | Test Accuracy |
|-|--------|---------------------------|---------------------|---------------|
|1| CNN v1 | Basic CNN, no Dropout     |       11.18%        |     45.11%    |
|2| CNN v2 |Dropout+EarlyStopping+stratified split| 51.46%   |     52.93%    |
|3| CNN v3 | ------------------------  |       ---           |     ---       |
|4| CNN v4 | ------------------------  |       ---           |     ---       |

### Experiment Notes

CNN v1 was used as a baseline model. The model achieved high training accuracy but poor validation performance, indicating severe overfitting.

CNN v2 improved generalization by introducing dropout regularization, early stopping, and a stratified train-validation split. The validation and test accuracies became more consistent, indicating a more reliable evaluation pipeline.