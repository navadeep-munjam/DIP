import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import argparse

from core.preprocess import load_image, denoise_image
from core.features import extract_features
from core.model import train_model, save_model

logging.basicConfig(level=logging.INFO)

def load_dataset(folder, label):
    X, y = [], []
    for f in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
        path = os.path.join(folder, f)
        try:
            img = load_image(path)
            noise = denoise_image(img)
            features = extract_features(noise)
            X.append(features)
            y.append(label)
        except (IOError, ValueError) as e:
            logging.warning(f"Error processing {f}: {e}")
    return np.array(X), np.array(y)

def main(scanner_path, camera_path, save_path):
    X1, y1 = load_dataset(scanner_path, 1)
    X2, y2 = load_dataset(camera_path, 0)

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])

    # Split the dataset for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Save the model and optionally the scaler
    save_model(model, save_path)
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scanner', default='images/denoised_scanner', help='Path to scanner images')
    parser.add_argument('--camera', default='images/denoised_camera', help='Path to camera images')
    parser.add_argument('--save', default='model.pkl', help='Path to save the trained model')
    args = parser.parse_args()

    main(args.scanner, args.camera, args.save)


# import os
# import numpy as np
# from core.preprocess import load_image, denoise_image
# from core.features import extract_features
# from core.model import train_model, save_model
# from tqdm import tqdm

# def load_dataset(folder, label):
#     X, y = [], []
#     for f in tqdm(os.listdir(folder), desc=f"Loading {folder}"):
#         path = os.path.join(folder, f)
#         try:
#             img = load_image(path)
#             noise = denoise_image(img)
#             features = extract_features(noise)
#             X.append(features)
#             y.append(label)
#         except Exception as e:
#             print(f"Error with {f}: {e}")
#     return np.array(X), np.array(y)

# if __name__ == "__main__":
#     X1, y1 = load_dataset("images/denoised_scanner", 1)
#     X2, y2 = load_dataset("images/denoised_camera", 0)

#     X = np.vstack([X1, X2])
    

#     X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#     y = np.concatenate([y1, y2])

#     model = train_model(X, y)
#     save_model(model)
#     print("Model trained and saved.")
