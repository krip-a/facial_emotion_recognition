import numpy as np

def preprocess(images, labels):
    pixel_images = []

    pixel_images = np.array([np.array(image) for image in images])

    """
    corresponds to the block:
        for image in images:
            pixel_images.append(np.array(image))    #convert each image to an array
        
        pixel_images = np.array(pixel_images)       #add each converted array to a big list
    """
    
    normalized_images = pixel_images / 255.0        #divide each pixel by 255, range of pixels 0-1 
    
    emotion_map = {
        "angry" : 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6
    }
    encoded_labels = np.array([emotion_map[label] for label in labels])

    """    
        encoded_labels = []
        for label in labels:
            encoded_labels.append(emotion_map[label])
        encoded_labels = np.array(encoded_labels)
    """

    images_dataset = normalized_images[:, :, :, np.newaxis]       #creating channel = 1 for grayscale
    #(number of images, height, width, channels) = (28709, 48, 48, 1)
    return images_dataset, encoded_labels


#test
if __name__ == "__main__":
    from load_data import load_images
    images, labels = load_images("data/train")
    X,y = preprocess(images, labels)

    print("Images shape: ", X.shape)
    print("Labels shape: ", y.shape)
    print("First Label: ", y[0])