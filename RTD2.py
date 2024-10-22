import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model  # Ensure you have the correct import for loading your model

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load your trained model
model = load_model('action.h5')  # Update the path to your model

# Define your class names based on your training data
class_names = ["Hello", "Goodbye", "Thank You"]  # Update this list based on your model training

def mediapipe_detection(image, model):
    """Convert BGR image to RGB and process it with the MediaPipe model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image color from BGR to RGB
    image.flags.writeable = False                  # Set the image to read-only for improved performance
    results = model.process(image)                 # Make predictions with MediaPipe
    image.flags.writeable = True                   # Set the image to writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the image back to BGR
    return image, results

def draw_landmarks(image, results):
    """Draw face, pose, and hand landmarks on the image."""
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def classify_pose(results):
    """Classify the pose based on detected landmarks."""
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks).flatten()
        landmarks = np.expand_dims(landmarks, axis=0)  # Add batch dimension

        # Ensure the landmarks have the correct shape
        if landmarks.shape[1] != 1662:
            landmarks = np.resize(landmarks, (1, 30, 1662))

        prediction = model.predict(landmarks)
        pose_index = np.argmax(prediction)

        # Return the corresponding class name instead of just the index
        return class_names[pose_index]  # Return class name based on prediction

    return "No Pose Detected"

def main():
    """Main function to run the pose detection application."""
    # Set up webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks on the frame
            draw_landmarks(image, results)

            # Classify the pose
            detected_pose = classify_pose(results)

            # Display the detected pose name on the frame
            cv2.putText(image, detected_pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Sign Language Detection', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
