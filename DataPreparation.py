import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split


def prepare_grayscale_dataset(dataset_path):
    """
    Prepare emotion dataset with 48x48 grayscale images

    Args:
        dataset_path (str): Root directory with emotion subdirectories

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']

    X = []
    y = []

    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(dataset_path, emotion)

        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)

            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Resize to 48x48
            img = cv2.resize(img, (48, 48))

            # Normalize pixel values
            img = img / 255.0

            X.append(img)
            y.append(emotion_idx)

    # Convert to numpy arrays
    X = np.array(X).reshape(-1, 48, 48, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=len(emotions))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test