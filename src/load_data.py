import os
from PIL import Image                   #Pillow

def load_images(data_path):             #data_path from main()

    images = []                        #pixels
    labels = []                        #emotions

    for emotion in sorted(os.listdir(data_path)):                   #get names of all emotion folders (angry, sad, etc.) 
        #print(f"loading {emotion}...")

        emotion_path = os.path.join(data_path, emotion)     #join data/train/ + happy/ for each emotion
        
        if os.path.isdir(emotion_path):

            for image_name in sorted(os.listdir(emotion_path)):
                image_path = os.path.join(emotion_path, image_name)

                try:
                    image = Image.open(image_path).convert("L")     #open the image as grayscale
                    image = image.resize((48,48))
                except Exception as e:
                    print(image_path, e)
                    continue

                images.append(image)
                labels.append(emotion)

                """
                image = PIL.Image.Image object
                images = list of PIL objects
                """

    return images, labels

"""
data_path:
    received from main()
    path to data/train/ or data/test/ 

Workflow
    loop through the train or test folders
    read the name of immediate children and label them emotions
    loop through every image in each emotion folder
    (form path for each image)
    open image in grayscale
    store the image and its label
    return two lists:
        images -> image objects
        labels -> emotion names
"""

#test
if __name__ == "__main__":
    images, labels = load_images("data/train")
    print("Nummber of images: ", len(images))
    print("First label:", labels[0])
    print("First image size:", images[0].size)
    