import os
import cv2
import numpy as np

IMG_SIZE = 150


def preprocess_image(image_path):
    """Load, resize, and normalize one chest X-ray image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0
    return image


def preview_samples(folder_path, limit=5):
    """Preprocess a few sample images to verify the Phase-1 pipeline."""
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    count = 0
    for filename in os.listdir(folder_path):
        if count >= limit:
            break
        image_path = os.path.join(folder_path, filename)
        image = preprocess_image(image_path)
        if image is not None:
            print(filename, image.shape, image.min(), image.max())
            count += 1


if __name__ == "__main__":
    preview_samples("dataset/Normal")
