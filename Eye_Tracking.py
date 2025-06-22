import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import winsound
import warnings

def initialize_models():
    return {
        "face_mesh": mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True),
    }

def get_landmark_points(landmarks, indexes, w, h):
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes])

def process_eye(frame, face_results):
    if not face_results.multi_face_landmarks:
        return ""
    
    h, w, _ = frame.shape
    face_landmarks = face_results.multi_face_landmarks[0]
    
    left_eye = get_landmark_points(face_landmarks.landmark, [33, 160, 158, 133, 153, 144], w, h)
    right_eye = get_landmark_points(face_landmarks.landmark, [263, 387, 385, 362, 380, 373], w, h)
    left_iris = get_landmark_points(face_landmarks.landmark, [468, 469, 470, 471, 472], w, h)
    right_iris = get_landmark_points(face_landmarks.landmark, [473, 474, 475, 476, 477], w, h)
    
    left_eye_center, right_eye_center = np.mean(left_eye, axis=0).astype(int), np.mean(right_eye, axis=0).astype(int)
    left_iris_center, right_iris_center = np.mean(left_iris, axis=0).astype(int), np.mean(right_iris, axis=0).astype(int)
    
    def get_eye_aspect_ratio(eye):
        return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2.0 * np.linalg.norm(eye[0] - eye[3]))
    threshold_right = 4
    threshold_left = 3
    if get_eye_aspect_ratio(left_eye) < 0.2 and get_eye_aspect_ratio(right_eye) < 0.2:
        return "BLINKING"
    if right_iris_center[0] > right_eye_center[0] + threshold_right:
        return "Eye Right"
    if right_iris_center[0] < right_eye_center[0] - threshold_left:
        return "Eye Left"
    return "Eye Center"

def main():  

    models = initialize_models()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        start = time.time()
        frame = cv2.flip(frame, 1)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = models["face_mesh"].process(processed_frame)
        
        eye_text = process_eye(frame, face_results)
        
        for i, text in enumerate([eye_text]):
            if text:
                cv2.putText(frame, text, (20, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, f'FPS: {int(1 / (time.time() - start))}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 176, 242), 2)
        cv2.imshow('AI Multimodal Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()