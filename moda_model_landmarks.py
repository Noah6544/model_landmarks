"""
ModaMate.py

This script, part of the ModaMate project, is designed to capture video from a webcam and 
extract 2D pose landmarks using MediaPipe's holistic solutions. It also allows for passing an image
and having the landmarks drawn on the image instead like the examples shown in the output folder

Author: Emmanuel Oben
Date: Nov 21st 2023
"""

import cv2
import mediapipe as mp
import pathlib
import os
import numpy as np

IMAGE_SOURCE_DIR = "images";
LANDMARK_DIR = "image_landmarks"

def initialize_media_pipe():
    """
    Initializes and returns MediaPipe Holistic object with specified confidence levels.
    """
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return holistic, mp_holistic

def initialize_drawing_utils():
    """
    Initializes and returns MediaPipe DrawingUtils.
    """
    return mp.solutions.drawing_utils

def process_image(image, results, mp_holistic, drawing_utils):
    """
    Draw pose landmarks on image.
    """

    # Draw pose landmarks
    if results.pose_landmarks:
        # Draw face landmarks
        drawing_utils.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=2),
            drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        
        # Draw left hand landmarks
        drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(80, 44, 255), thickness=2, circle_radius=2)
        )
        
        # Draw right hand landmarks
        drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        
        # Pose detection
        drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 8, 8), cv2.FILLED)
            
        # Convert the image color back from RGB to BGR
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def capture_from_webcam(holistic, mp_holistic, drawing_utils):
    """
    Captures video from the webcam and processes each frame.
    """
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image color from BGR to RGB
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = holistic.process(image)
        
        process_image(image, results, mp_holistic, drawing_utils)

        cv2.imshow('OpenPose Demo', image)
        
        if cv2.waitKey(5) & 0xFF == 27:     # Exit on 'Esc' key
            break
        if cv2.getWindowProperty('OpenPose Demo', cv2.WND_PROP_VISIBLE) < 1:  # Check if window is closed
            break

    cap.release()
    cv2.destroyAllWindows()
    
def draw_landmarks_on_black(image, results, mp_holistic, drawing_utils):
    """
    Draws pose landmarks on a black background image.
    """

    # Create a black image of the same size as the original
    h, w, c = image.shape
    black_image = np.zeros((h, w, c), dtype=np.uint8)

    #process_image(black_image, holistic, mp_holistic, drawing_utils);
    
    """
    Processes an image to detect and draw pose landmarks.
    """
    # Convert the image color from BGR to RGB
    cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)

    return process_image(black_image, results, mp_holistic, drawing_utils)

    #return black_image
    
def save_image(image_path, image, results, mp_holistic, drawing_utils):
    results_dir = LANDMARK_DIR
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Extract the filename and append '_landmark'
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_landmark{ext}"

    # Draw landmarks on black background and save
    black_landmark_image = draw_landmarks_on_black(image, results, mp_holistic, drawing_utils)
    save_path = os.path.join(results_dir, new_filename)
    cv2.imwrite(save_path, black_landmark_image)

    print(f"Saved: {save_path}")

def main():
    """
    Main function to choose the mode of operation: Capture from webcam or process a given image.
    """
    holistic, mp_holistic = initialize_media_pipe()
    drawing_utils = initialize_drawing_utils()

    mode = input("Enter 'C' to capture from webcam or 'I' to process an image: ").strip().upper()
    if mode == 'C':
        capture_from_webcam(holistic, mp_holistic, drawing_utils)
    elif mode == 'I':
        image_name = input("Enter the name of the image (Image should be in the images folder): ")
        image_path = os.path.join(IMAGE_SOURCE_DIR, image_name)
        image = cv2.imread(image_path);
        
        if image is not None:
            # Convert the image color from BGR to RGB
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = holistic.process(image)
            
            process_image(image, results, mp_holistic, drawing_utils)
            cv2.imshow("Processed Image", image)
            
            save_image(image_path, image, results, mp_holistic, drawing_utils);
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to load the image.")
    else:
        print("Invalid input.")

if __name__ == "__main__":
    main()