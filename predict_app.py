from src.predict import predict_image

image_path = input("Enter image path: ")

emotion = predict_image(
    "saved_models/emotion_cnn.keras",
    image_path
)

print("Predicted Emotion: ", emotion)