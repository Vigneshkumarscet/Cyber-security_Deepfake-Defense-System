import cv2
import numpy as np
from imutils import face_utils
import dlib

# Load pre-trained face detection model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to extract facial landmarks
def extract_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape

# Function to calculate Euclidean distances between consecutive landmarks
def calculate_distances(landmarks):
    distances = []
    for i in range(1, len(landmarks)):
        distance = np.linalg.norm(landmarks[i] - landmarks[i - 1])
        distances.append(distance)
    return distances

# Function to detect anomalies in facial landmark distances
def detect_anomalies(distances):
    # Example anomaly detection logic (you can replace with more sophisticated techniques)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 2 * std_distance
    anomalies = [dist for dist in distances if dist > threshold]
    return anomalies

# Function to process video and detect deepfakes
def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    suspicious_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Extract facial landmarks
        landmarks = extract_landmarks(frame)

        if landmarks is not None:
            # Calculate distances between facial landmarks
            distances = calculate_distances(landmarks)
            
            # Detect anomalies in distances
            anomalies = detect_anomalies(distances)
            
            # If anomalies are detected, mark the frame as suspicious
            if anomalies:
                suspicious_frames += 1
                print("Suspicious frame detected at frame:", frame_count)

    cap.release()
    suspiciousness_ratio = suspicious_frames / frame_count

    # If suspiciousness ratio is above a certain threshold, consider the video as deepfake
    if suspiciousness_ratio > 0.1:  # Adjust the threshold as per requirements
        print("Deepfake detected!")
    else:
        print("Video seems authentic.")

# Example usage
video_path = "path_to_your_video.mp4"
detect_deepfake(video_path)
