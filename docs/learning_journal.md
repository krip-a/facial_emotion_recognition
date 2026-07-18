
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
