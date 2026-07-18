#define the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense
)

def create_cnn():
    model = Sequential()    #Run the layers one after another

    #First Convolution layer
    #Convolution: learn local visual features (edges, curves, corners, eyess, etc.)
    #               from small regions of the images
    #Multiple Conv2D layers => enough information to classify whole face
    model.add(
        Conv2D(
            filters = 32,               #32 feature detectors
            kernel_size = (3,3),        #slide tiny 3x3 windows all over the img
            activation = "relu",        #rectified linaer unit; introduces non-linearity
            input_shape = (48, 48, 1)
        )
    )

    #Pooling: keep the strongest singnal and reduce dimension
    model.add(
        MaxPooling2D(pool_size = (2,2))
    )

    """ Stacking makes the model stronger, so repeat"""
    #Second Convolution layer
    model.add(
        Conv2D(
            filters = 64,               #64 feature detectors
            kernel_size = (3,3),        
            activation = "relu",        
        )
    )

    #Second Pooling
    model.add(
        MaxPooling2D(pool_size = (2,2))
    )

    #Flatten: convert the multi-dimensional outputs to one-dimensional vector
    #Prepare the data for dense layers
    model.add(
        Flatten()
    )

    #Classifier
    #Dense Layer: combines everything learned
    #expects a flat vector
    model.add(
        Dense(
            128,
            activation = "relu"
        )
    )

    #Output layer
    model.add(
        Dense(
            7,                      #because 7 emotions
            activation = "softmax"  #converts outputs into probabilities
        )
    )

    #Compile
    model.compile(
        optimizer = "adam",          #how network updates weights, standard choice
        loss="sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model