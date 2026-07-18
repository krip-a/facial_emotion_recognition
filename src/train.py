from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def train_model(model, train_images, train_labels):

    """
    Args:
        model: compiled CNN model
        train_images: preprocessed training images
        train_labels: encoded training labels
    Returns:
        model: trained model
        history: training history returned by model.fit
    """
    early_stop = EarlyStopping(         #watches during training and stops automatically if the model stops improving
        monitor = "val_loss",           #watch the 'val_loss' while training
        patience = 5,                   #wait 5 epochs (if not improving) before stopping
        restore_best_weights = True     #choose the epoch with the best monitored metric (val_loss here)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size = 0.2,
        random_state = 42,
        stratify = train_labels         #keep the same proportion of emotion distribution in traiing and validation
    )

    history = model.fit(
        X_train,
        y_train,
        epochs = 30,                #let the model see the entire training dataset 30 times
        batch_size = 32,            #process 32 images at a time(rather than all at once), compute loss, update, continue 
        #validation_split = 0.2,     #20% data for validation; monitor performance on unseen data for each epoch
        validation_data = (X_val, y_val),
        shuffle = True,             #shuffle before each epoch so order doesn't matter
        callbacks = [early_stop],    #stop around 8-10 epochs if the model stops improving
        verbose = 1                 #shows the training process
    )
    return model, history

"""
    validation_split=0.2 takes the last 20% of the data for validation
    but out images load one emotion after another
    so the last 20% will always belong to the same emotion
    there is no variation in emotion, so the validation accuracy is always low

    better practice is to split it by yourself
"""
"""
    training_loss = how wrong the model is on images it has already seen
    validation_loss = how wrong the model is on images it has never seen

    if train_loss improves but val_loss decreases, it means the model is memorizing
"""