import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, GaussianBlur, CLAHE, RandomBrightnessContrast, Compose
)
from albumentations.core.composition import OneOf

class CustomEmotionModel:
    def __init__(self, emotions=['happy', 'sad', 'angry', 'surprised', 'neutral']):
        self.emotions = emotions
        self.model = self.build_emotion_cnn()

    def build_emotion_cnn(self):
        """Build Convolutional Neural Network for emotion recognition"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.emotions), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_dataset(self, dataset_path, visualize=True):
        """
        Prepare emotion dataset with face detection and data augmentation

        Args:
            dataset_path (str): Root directory with emotion subdirectories
            visualize (bool): Whether to print example images

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Define augmentation pipeline using Albumentations
        augmentation = Compose([
            HorizontalFlip(p=0.5),  # 50% chance to flip horizontally
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            GaussianBlur(blur_limit=(3, 5), p=0.3),  # Slight blurring
            OneOf([
                CLAHE(clip_limit=2),
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
            ], p=0.5)  # Enhance contrast/brightness
        ])

        X = []
        y = []

        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_path = os.path.join(dataset_path, emotion)

            for image_file in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_file)

                try:
                    # Read image
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    # Process first detected face
                    if len(faces) > 0:
                        (x, y_coord, w, h) = faces[0]

                        # Crop face
                        face = gray[y_coord:y_coord + h, x:x + w]

                        # Resize to 48x48
                        face = cv2.resize(face, (48, 48))

                        # Original face
                        X.append(face / 255.0)
                        y.append(emotion_idx)

                        # Generate augmented images (5 per original image)
                        for _ in range(5):
                            # Augment the face using Albumentations
                            augmented_face = augmentation(image=face)['image']

                            # Normalize augmented face
                            augmented_face = augmented_face / 255.0

                            # Optional visualization
                            if visualize and len(X) < 15:
                                plt.figure(figsize=(10, 5))

                                # Original face
                                plt.subplot(1, 2, 1)
                                plt.imshow(face, cmap='gray')
                                plt.title(f"Original: {image_file}")
                                plt.axis('off')

                                # Augmented face
                                plt.subplot(1, 2, 2)
                                plt.imshow(augmented_face, cmap='gray')
                                plt.title(f"{emotion} - Augmented")
                                plt.axis('off')

                                plt.tight_layout()
                                plt.show()

                            # Add augmented image
                            X.append(augmented_face)
                            y.append(emotion_idx)

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # One-hot encode labels
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.emotions))

        # Reshape X for model input
        X = X.reshape(-1, 48, 48, 1)

        # Split into training and testing sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, dataset_path, epochs=50):
        """Train emotion recognition model"""
        X_train, X_test, y_train, y_test = self.prepare_dataset(dataset_path)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            verbose=1
        )

        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return history

    def predict_emotion(self, face_image):
        """Predict emotion from a face image"""
        # Preprocess image
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        normalized_face = resized_face / 255.0
        input_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotion
        prediction = self.model.predict(input_face)
        emotion_index = np.argmax(prediction)
        return self.emotions[emotion_index]

class EmojiOverlay:
    def __init__(self, emoji_folder="EmotionEmoji"):
        self.emoji_map = {}
        self.frame_index = 0
        self.load_emojis(emoji_folder)

    def load_emojis(self, emoji_folder):
        """Load all emoji GIFs from the specified folder"""
        emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral']

        for emotion in emotions:
            gif_path = os.path.join(emoji_folder, f"{emotion}Emoji.gif")
            try:
                if os.path.exists(gif_path):
                    self.emoji_map[emotion] = self.load_gif_frames(gif_path)
                else:
                    print(f"Warning: Emoji file not found: {gif_path}")
            except Exception as e:
                print(f"Error loading {emotion} emoji: {e}")

    def load_gif_frames(self, gif_path):
        """Load and extract frames from a GIF file"""
        frames = []
        try:
            gif = Image.open(gif_path)
            for frame in range(gif.n_frames):
                gif.seek(frame)
                frame_image = np.array(gif.convert("RGBA"))
                frames.append(frame_image)
        except Exception as e:
            print(f"Error processing GIF {gif_path}: {e}")
        return frames

    def overlay_emoji(self, frame, face_coords, emotion):
        """Overlay emoji on the frame at specified coordinates"""
        if emotion not in self.emoji_map or not self.emoji_map[emotion]:
            return frame

        emoji_frames = self.emoji_map[emotion]
        emoji = emoji_frames[self.frame_index % len(emoji_frames)]

        # Extract face coordinates
        x, y, w, h = face_coords

        # Position emoji above the face
        overlay_y = max(0, y - h - 10)
        overlay_x = x

        try:
            # Resize emoji to match face width
            emoji = cv2.resize(emoji, (w, h))

            # Create a region of interest (ROI)
            roi = frame[overlay_y:overlay_y + h, overlay_x:overlay_x + w]

            # Check if ROI dimensions match emoji dimensions
            if roi.shape[:2] != emoji.shape[:2]:
                return frame

            # Apply alpha blending
            alpha = emoji[:, :, 3] / 255.0
            for c in range(3):
                frame[overlay_y:overlay_y + h, overlay_x:overlay_x + w, c] = \
                    (emoji[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8)

        except Exception as e:
            print(f"Error overlaying emoji: {e}")

        self.frame_index += 1
        return frame

def main():
    # Initialize custom emotion model
    emotion_model = CustomEmotionModel()

    # Optional: Train model with your custom dataset
    emotion_model.train_model('/Users/jovany/PycharmProjects/FaceDetectionSystem/Dataset')

    # Initialize emoji overlay handler
    emoji_handler = EmojiOverlay()

    # Initialize video capture
    camera_index = 1
    video_capture = cv2.VideoCapture(camera_index)

    if not video_capture.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Get face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        try:
            frame = cv2.resize(frame, (360, 240))
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract face region
                face_roi = frame[y:y + h, x:x + w]

                # Predict emotion using custom model
                emotion = emotion_model.predict_emotion(face_roi)

                # Display emotion text
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Overlay emoji
                frame = emoji_handler.overlay_emoji(frame, (x, y, w, h), emotion)

        except Exception as e:
            print(f"Error in main loop: {e}")

        cv2.imshow('Emotion Detection with Emoji Overlay', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()