import joblib
import cv2
import numpy as np

from core.preprocess import denoise_image
from core.features import extract_features

# 1. Load trained pipeline
pipeline = joblib.load('model.pkl')

# 2. Full preprocessing pipeline
def preprocess_and_extract(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)

    img_denoised = denoise_image(img)
    features = extract_features(img_denoised)  # Shape: (19,) or whatever feature vector you trained on

    return np.array(features).reshape(1, -1)  # Shape: (1, n_features)

# 3. Predict
img_processed = preprocess_and_extract("/home/navadeep/Desktop/DIP/images/denoised_scanner/denoised_scanner_16.png")
prediction = pipeline.predict(img_processed)[0]
print("Predicted:", "Scanner" if prediction == 1 else "Camera")


# import joblib
# import cv2
# import numpy as np

# # 1. Load the FULL pipeline
# pipeline = joblib.load('model.joblib')

# # 2. Preprocess EXACTLY like training
# def preprocess_image(image_path, target_size=(64, 64)):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Must match training
#     img = cv2.resize(img, target_size)
#     img = img.reshape(1, -1)  # Critical: shape (1, 4096) not (4096,)
#     return img

# # 3. Predict
# img_processed = preprocess_image("/home/navadeep/Desktop/DIP/test_images/scanner_77.png")
# prediction = pipeline.predict(img_processed)[0]  # No extra [] needed now
# print("Predicted:", prediction)

# import joblib
# import cv2
# import numpy as np
# import os

# # Load the model (same directory)
# model = joblib.load('model.joblib')

# # Preprocess function (must match training)
# def preprocess_image(image_path, target_size=(64, 64)):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Adjust if training used color
#     img = cv2.resize(img, target_size)
#     img = img / 255.0  # Normalize if training did this
#     return img.flatten()  # Flatten for SVM

# test_dir = "test_images"  # Folder in the same directory

# # Predict and print results
# for img_file in os.listdir(test_dir):
#     img_path = os.path.join(test_dir, img_file)
#     img_processed = preprocess_image(img_path)
#     prediction = model.predict([img_processed])[0]
#     class_name = "scanner" if prediction == 0 else "camera"
#     print(f"{img_file}: {class_name} (Class {prediction})")

