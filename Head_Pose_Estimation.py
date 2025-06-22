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
        "face_detection": mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    }

def get_landmark_points(landmarks, indexes, w, h):
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes])

def estimate_head_pose(face_results, frame):
    if not face_results.multi_face_landmarks:
        return ""
    
    img_h, img_w, _ = frame.shape
    face_landmarks = face_results.multi_face_landmarks[0]
    
    face_2d = np.array([(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h)) for i in [33, 263, 1, 61, 291, 199]], dtype=np.float64)
    face_3d = np.array([(int(face_landmarks.landmark[i].x * img_w), int(face_landmarks.landmark[i].y * img_h), face_landmarks.landmark[i].z) for i in [33, 263, 1, 61, 291, 199]], dtype=np.float64)
    
    cam_matrix = np.array([[img_w, 0, img_w / 2], [0, img_w, img_h / 2], [0, 0, 1]])
    _, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1), dtype=np.float64))
    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    x, y = angles[0] * 360, angles[1] * 360
    
    return "Looking Left" if y < -10 else "Looking Right" if y > 10 else "Looking Down" if x < -10 else "Looking Up" if x > 10 else "Looking Forward"

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
        
        head_pose_text = estimate_head_pose(face_results, frame)
        
        for i, text in enumerate([head_pose_text]):
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