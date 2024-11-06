# Extracting Frames from Dataset
import cv2
import os
import re

def sanitize_folder_name(folder_name):
    # Replace any invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', folder_name)

def extract_frames_from_videos(video_folder, output_folder, fps=5):
    # Ensure base output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all video files in the directory and its subdirectories
    video_files = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other extensions if needed
                video_files.append(os.path.join(root, file))

    # Check if any video files were found
    if not video_files:
        print(f"No video files found in {video_folder}.")
        return

    # Loop through all videos
    for video_file in video_files:
        video_path = video_file
        
        # Creating a subfolder for each video to store its frames
        video_name = sanitize_folder_name(os.path.splitext(os.path.basename(video_file))[0])
        video_output_folder = os.path.join(output_folder, video_name)

        # Create the output folder for the video, including parent directories if necessary
        try:
            os.makedirs(video_output_folder, exist_ok=True)
        except FileNotFoundError as e:
            print(f"Error creating directory {video_output_folder}: {e}")
            continue

        # Extract frames from each video
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video
        
        # Check if frame_rate is valid
        if frame_rate == 0 or frame_rate is None:
            print(f"Warning: Could not determine frame rate for {video_file}. Skipping...")
            cap.release()
            continue
        
        success, image = cap.read()
        
        while success:
            if count % int(frame_rate / fps) == 0:
                # Save frame as an image
                frame_filename = os.path.join(video_output_folder, f"frame_{count}.jpg")
                cv2.imwrite(frame_filename, image)
                print(f"Saved frame {count} from {video_file}")
                
            success, image = cap.read()
            count += 1

        cap.release()
        print(f"Finished extracting frames for {video_file}")

# Example usage:
video_folder = "C:\\SignLanguage\\Dataset"
output_folder = "C:\\SignLanguage\\OutputFolder"
extract_frames_from_videos(video_folder, output_folder, fps=5)

# Data Preprocessing
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_images(image_folder, output_folder, target_size=(224, 224), augment=False):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Set up data augmentation if required
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    # Loop through each class folder in the image folder
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

            # Process each image in the class folder
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)

                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Resize the image
                image = cv2.resize(image, target_size)
                
                # Normalize the image
                image = image / 255.0  # Scale pixel values to [0, 1]

                # Save the original processed image
                output_image_path = os.path.join(output_folder, class_name, image_name)
                cv2.imwrite(output_image_path, (image * 255).astype(np.uint8))  # Convert back to [0, 255] for saving

                # If augmentation is enabled, apply transformations
                if augment:
                    # Reshape image for data generator
                    image = np.expand_dims(image, axis=0)

                    # Generate augmented images
                    for i, augmented_image in enumerate(datagen.flow(image, batch_size=1)):
                        aug_image_path = os.path.join(output_folder, class_name, f"{image_name.split('.')[0]}_aug_{i}.jpg")
                        cv2.imwrite(aug_image_path, (augmented_image[0] * 255).astype(np.uint8))
                        if i >= 5:  # Save only 5 augmented images
                            break

# Example usage
image_folder = "C:\\SignLanguage\\OutputFolder"  # Folder with extracted frames
output_folder = "C:\\SignLanguage\\ProcessedDataset"  # Folder for processed images
preprocess_images(image_folder, output_folder, target_size=(224, 224), augment=True)

# Checking Data labels
import os
from PIL import Image

def check_labels(image_folder):
    # Loop through each class folder in the image folder
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            print(f"Class: {class_name}")
            # Loop through each image in the class folder
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                # Load and display the image
                try:
                    img = Image.open(image_path)
                    img.show()  # This will open the image in the default viewer
                    print(f"Image: {image_name} - Label: {class_name}")
                except Exception as e:
                    print(f"Could not open image {image_name}: {e}")

# Example usage
image_folder = "C:\\SignLanguage\\ProcessedDataset"  # Your folder with processed images
check_labels(image_folder)

import os
import shutil
import random

def split_dataset(image_folder, output_folder, train_ratio=0.7, val_ratio=0.15):
    # Create output directories for train, val, and test sets
    for split in ['train', 'val', 'test']:
        split_folder = os.path.join(output_folder, split)
        os.makedirs(split_folder, exist_ok=True)

    # Loop through each class folder
    for class_name in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            images = os.listdir(class_folder)
            random.shuffle(images)

            # Calculate the number of images for each split
            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * val_ratio)

            train_images = images[:train_split]
            val_images = images[train_split:train_split + val_split]
            test_images = images[train_split + val_split:]

            # Copy images to respective folders
            for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
                split_class_folder = os.path.join(output_folder, split, class_name)
                os.makedirs(split_class_folder, exist_ok=True)
                for image_name in split_images:
                    shutil.copy(os.path.join(class_folder, image_name), os.path.join(split_class_folder, image_name))

# Example usage:
image_folder = "C:\\SignLanguage\\ProcessedDataset"
output_folder = "C:\\SignLanguage\\FinalDataset"
split_dataset(image_folder, output_folder)


# resnet50 Architecture 
# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Define the ResNet50 model using transfer learning
def create_resnet50_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model (pre-trained weights)
    base_model.trainable = False

    # Create a new model on top of it
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# Example usage:
input_shape = (224, 224, 3)  # Assuming your images are 224x224 RGB
num_classes = 10  # Set the number of classes based on your dataset

# Create the model
model_resnet50 = create_resnet50_model(input_shape, num_classes)

# Compile the model
model_resnet50.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators for training, validation, and test sets
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize images
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set paths to the dataset folders
train_dir = "C:\\SignLanguage\\FinalDataset\\train"
val_dir = "C:\\SignLanguage\\FinalDataset\\val"
test_dir = "C:\\SignLanguage\\FinalDataset\\test"

# Load the datasets from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,  # Adjust based on memory limitations
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

