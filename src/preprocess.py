import numpy as np

def preprocess(images, labels):
    pixel_images = []

    pixel_images = np.array([np.array(image) for image in images])

    """
    corresponds to the block:
        for image in images:
            pixel_images.append(np.array(image))
        
        pixel_images = np.array(pixel_images)
    """
    
    normalized_images = pixel_images / 255.0        #divide each pixel by 255 
    
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
    images_dataset = normalized_images[:, :, np.newaxis]

    return images_dataset, encoded_labels