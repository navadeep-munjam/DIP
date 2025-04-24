import numpy as np
import cv2
import os

def generate_scanner_image(shape=(256, 256), periodic_strength=10, save_path=None):
    image = np.random.normal(128, 10, size=shape)
    periodic_pattern = np.sin(np.linspace(0, 10 * np.pi, shape[1])) * periodic_strength
    for i in range(shape[0]):
        image[i, :] += periodic_pattern
    image = np.clip(image, 0, 255).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, image)

def generate_camera_image(shape=(256, 256), save_path=None):
    noise = np.random.normal(128, 10, size=shape)
    high_freq = np.random.normal(0, 5, size=shape)
    image = noise + cv2.GaussianBlur(high_freq, (3, 3), 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if save_path:
        cv2.imwrite(save_path, image)

def create_dataset(n_per_class=30, output_dir="images"):
    os.makedirs(f"{output_dir}/scanner", exist_ok=True)
    os.makedirs(f"{output_dir}/camera", exist_ok=True)

    print(f"Generating {n_per_class} scanner images...")
    for i in range(n_per_class):
        generate_scanner_image(save_path=f"{output_dir}/scanner/scanner_{i}.png")

    print(f"Generating {n_per_class} camera images...")
    for i in range(n_per_class):
        generate_camera_image(save_path=f"{output_dir}/camera/camera_{i}.png")

    print("✅ Synthetic dataset generated successfully.")

if __name__ == "__main__":
    create_dataset(n_per_class=100)
