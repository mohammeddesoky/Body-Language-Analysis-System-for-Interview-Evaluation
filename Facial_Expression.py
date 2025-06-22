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
    device = "cuda"
    model_name = get_model_list()[0]
    return {
        "fer": EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device),
        "face_detection": mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    }

def detect_emotion(fer, frame, face_detection):
    results = face_detection.process(frame)
    if results.detections:
        bboxC = results.detections[0].location_data.relative_bounding_box
        x1, y1, x2, y2 = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0]), int((bboxC.xmin + bboxC.width) * frame.shape[1]), int((bboxC.ymin + bboxC.height) * frame.shape[0])
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return fer.predict_emotions([face], logits=True)[0][0]
    return ""

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
        
        emotion = detect_emotion(models["fer"], processed_frame, models["face_detection"])
        
        for i, text in enumerate([f"Emotion: {emotion}"]):
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