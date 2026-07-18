import os

def save_model(model, save_path="saved_models/emotion_cnn.keras"):

    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    model.save(save_path)

    print(f"Model saved to {save_path}")