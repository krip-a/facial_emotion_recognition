from src.load_data import load_images
from src.preprocess import preprocess
from models.cnn import create_cnn
from src.train import train_model
from src.save_model import save_model
from src.evaluate import evaluate_model

#training data
train_images1, train_labels1 = load_images("data/train")
train_images, train_labels = preprocess(train_images1, train_labels1)

#create CNN model
model = create_cnn()

#Train CNN
model, history = train_model(model, 
                             train_images, 
                             train_labels)

#Save model
save_model(model)

#test_data
test_images1, test_labels1 = load_images("data/test")
test_images, test_labels = preprocess(test_images1, test_labels1)

#evaluate
evaluate_model(
    model, 
    test_images,
    test_labels
)