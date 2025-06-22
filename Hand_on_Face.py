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
        "hand_detector": mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5),
        "face_detection": mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    }

def get_landmark_points(landmarks, indexes, w, h):
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes])

def detect_hand_on_face(face_results, hand_results, frame):
    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            face_points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for hand_point in hand_landmarks.landmark:
                    hx, hy = int(hand_point.x * frame.shape[1]), int(hand_point.y * frame.shape[0])
                    if any(abs(hx - fx) < 30 and abs(hy - fy) < 30 for fx, fy in face_points):
                        return True
    return False

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
        hand_results = models["hand_detector"].process(processed_frame)

        hand_on_face = detect_hand_on_face(face_results, hand_results, frame)
        
        for i, text in enumerate([f"Hand on Face" if hand_on_face else ""]):
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