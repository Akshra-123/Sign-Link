# Import and Dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import mediapipe as mp

# Key point detection using media pipe
mp_holistic = mp.solutions.holistic # Holistic model this is for detecting the key points
mp_drawing = mp.solutions.drawing_utils # Drawing utilities this is for drawing those landmarks

def mediapipe_detection(image, model):   # image is the input image to be processed and model is mediapipe model used for detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction (uses the provided MediaPipe model to analyze the input image and return predictions)
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
