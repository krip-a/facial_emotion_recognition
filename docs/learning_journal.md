
## Issues Encountered, Improvements, and Lessons Learned

#### Issue 1: Severe Overfitting and Model Memorization
##### First Attempt 
Initial CNN Architechture:
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
Problem:
- The model learned the training data extremely well but failed to generalize to unseen data images
- The large gap between training and validation performance indicated overfitting (model learned not only the underlying patterns but also a lot of noise)
- The model was not learning general facial emotion patterns; instead, it was memorizing patterns specific to the training images, including noise and irrelevant details
Cause:
The initial architechture had:
- Multiple dense layers after flattening
- A large number of parameters after flattening + dense
- No regularization techniques
The final layers had enough capacity to memorize the training dataset

Changes Made:
- Added Dropout(0.5) before the output layer
- Added EarlyStopping callback
- Increased maximum epochs from 20 to 30 to allow the model more opportunity to learn while preventing unnecessary training

Early Stopping COnfiguration:
```python 
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
```
Result:
The model stopped training when validation performance stopped improving and restored the best-performing weights.