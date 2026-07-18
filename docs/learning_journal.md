
## Issues Encountered, Improvements, and Lessons Learned

#### Record of Experiments

|#| Model  |      Training Changes     | Validation Accuracy | Test Accuracy |
|-|--------|---------------------------|---------------------|---------------|
|1| CNN v1 | Basic CNN, no Dropout     |       11.18%        |     45.11%    |
|2| CNN v2 |Dropout+EarlyStopping+stratified split| 51.46%   |     52.93%    |
|3| CNN v3 | ------------------------  |       ---           |     ---       |
|4| CNN v4 | ------------------------  |       ---           |     ---       |

### CNN_version_1
### Issue 1: Severe Overfitting and Model Memorization
#### First Attempt 
##### Initial CNN Architechture:
```text 
(Conv2D + MaxPooling2D) x2 
Flatten
Dense 
Output Layer
```

The first model achieved:
```text
Epoch 20/20
    Training Accuracy:     97.12%
    Training Loss:         0.0940

    Validation Accuracy:   11.18%
    Validation Loss:       28.2124
Overall:
Test Accuracy:         45.11%
Test Loss:             8.6681
```
##### Problem:
- The model learned the training data extremely well but failed to generalize to unseen data images
- The large gap between training and validation performance indicated overfitting (model learned not only the underlying patterns but also a lot of noise)
- The model was not learning general facial emotion patterns; instead, it was memorizing patterns specific to the training images, including noise and irrelevant details

##### Cause:
The initial architechture had:
- Multiple dense layers after flattening
- A large number of parameters after flattening + dense
- No regularization techniques

The final layers had enough capacity to memorize the training dataset

##### Changes Made:
- Added Dropout(0.5) before the output layer
- Added EarlyStopping callback
- Increased maximum epochs from 20 to 30 to allow the model more opportunity to learn while preventing unnecessary training

##### Early Stopping Configuration:
```python 
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
```
##### Result:
The model stopped training when validation performance stopped improving and restored the best-performing weights.

### Issue 2: Incorrect Validation Split Caused Misleading Results
#### First Attempt 

##### The first training approach used:
```python
model.fit(
        train_images,
        train_labels,
        epochs = 20,               
        batch_size = 32,           
        validation_split = 0.2,    # <---- Here
        shuffle = True,              
        verbose = 1                
    )
```

##### Problem:
Keras' validation_split=0.2 takes the last 20% of the data as validation data before training. However, the dataset was loaded emotion-by-emotion:
```text
    train/
    |
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```
Since images were loaded sequentially by emotion category, the validation set was not randomly sampled and did not maintain the original class distribution
Therefore, the validation set was not representative of the overall dataset distribution.
This caused unreliable validation results.

##### Changes Made:
Replaced Keras' automatic validation split with an explicit stratified split:
```python
    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size = 0.2,
        random_state = 42,
        stratify = train_labels         
    )
```
This ensured:

- 80% of data used for training
- 20% used for validation
- All emotion classes maintained similar proportions in both sets

##### Training was changed to:
```python
    model.fit(
        X_train,
        y_train,
        epochs = 30,                
        batch_size = 32,            
        validation_data = (X_val, y_val),
        shuffle = True,             
        callbacks = [early_stop],    
        verbose = 1                 
    )
```

##### Result:
The validation and test performance became consistent.

```text
Epoch 14/30
    Training Accuracy:     63.24%  
    Training Loss:         0.949

    Validation Accuracy:   51.46%
    Validation Loss:       1.344
Overall:
Test Accuracy:             52.93%
Test Loss:                 1.247
```

### Issue 3: Validation Leakage and Misleading Evaluation Metrics

##### Problem:
The initial runs produced confusing results:

Example:
```text
Validation Accuracy: 92.39%
Test Accuracy: 53.25%
```
The large difference suggested that the validation score was not measuring the true generalization

##### Cause:
The model had already seen the validation images during training because the full dataset was used for training. Hence, the validation metric was overly optimistic. 

##### Solution:
Separated the datasets properly, using train_test_split from scikit-learn rather than using inbuilt validation_split from keras. 

##### Lesson Learned:
Validation accuracy is only meaningful when validation data is completely separated from training data. 

A high validation score does not always mean a good model; the data pipeline must also be correct. 


### CNN_version_2
Future Improvements loading:
- Augmentation
- Confusion Matrix
- Classification Report
- Choose and save the best model
