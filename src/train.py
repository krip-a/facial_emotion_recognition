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

    history = model.fit(
        train_images,
        train_labels,
        epochs = 20,                #let the model see the entire training dataset 20 times
        batch_size = 32,            #process 32 images at a time(rather than all at once), compute loss, update, continue 
        validation_split = 0.2,     #20% data for validation; monitor performance on unseen data for each epoch
        shuffle = True,             #shuffle before each epoch so order doesn't matter
        verbose = 1                 #shows the training process
    )
    return model, history

