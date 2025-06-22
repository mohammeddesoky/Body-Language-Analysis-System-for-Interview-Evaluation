import cv2
import numpy as np
import pandas as pd
import time
import mediapipe as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Models
def initialize_models():
    device = "cuda"
    model_name = get_model_list()[0]
    return {
        "fer": EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device),
        "face_mesh": mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True),
        "hand_detector": mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5),
        "face_detection": mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    }

# Landmark points
def get_landmark_points(landmarks, indexes, w, h):
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes])

# Detect Eyes Movement
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

# Detect Emotion
def detect_emotion(fer, frame, face_detection):
    results = face_detection.process(frame)
    if results.detections:
        bboxC = results.detections[0].location_data.relative_bounding_box
        x1, y1, x2, y2 = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0]), int((bboxC.xmin + bboxC.width) * frame.shape[1]), int((bboxC.ymin + bboxC.height) * frame.shape[0])
        face = frame[y1:y2, x1:x2]
        if face.size > 0:
            return fer.predict_emotions([face], logits=True)[0][0]
    return ""

# Detect Hand on Face
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

# Detect Faces Movement
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

# Alarm
def alarm(delay, time_, hand_on_face, head_pose_text, eye_text, emotion):
    
    if hand_on_face == False:
        time_[0] = time.time()
        delay[0] = 30
    if time.time() - time_[0] >= delay[0]:
        print(f"Hand on face detected for {delay[0]//60}:{delay[0]%60} Minute.")
        delay[0] += 15
        
    if head_pose_text == "Looking Forward": 
        time_[1] = time.time()
        delay[1] = 30
    if time.time() - time_[1] >= delay[1]:
        print(f"No Head movement detected for {delay[1]//60}:{delay[1]%60} Minute.")
        delay[1] += 15

    if eye_text == "Eye Center": 
        time_[2] = time.time()
        delay[2] = 30
    if time.time() - time_[2] >= delay[2]:
        print(f"No Eye Center detected for {delay[2]//60}:{delay[2]%60} Minute.")
        delay[2] += 15

    if emotion in ['Neutral', 'Happiness']: 
        time_[3] = time.time()
        delay[3] = 30
    if time.time() - time_[3] >= delay[3]:
        print(f"No emotion detected for {delay[3]//60}:{delay[3]%60} Minute.")
        delay[3] += 15

    return time_

# Main
def main(frame_skip=5):  
    
    frame_count = 0
    frame_numbers = []
    second = time.time()

    hand_on_face_history = []
    eye_text_history = []
    head_pose_text_history = []
    emotion_history = []
    seconds = []

    time_ = [time.time(), time.time(), time.time(), time.time()]
    delay = [30, 30, 30, 30]

    models = initialize_models()
    cap = cv2.VideoCapture(1)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        if frame_count % frame_skip == 0:
            continue

        start = time.time()
        frame = cv2.flip(frame, 1)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = models["face_mesh"].process(processed_frame)
        hand_results = models["hand_detector"].process(processed_frame)
        
        eye_text = process_eye(frame, face_results)
        emotion = detect_emotion(models["fer"], processed_frame, models["face_detection"])
        head_pose_text = estimate_head_pose(face_results, frame)
        hand_on_face = detect_hand_on_face(face_results, hand_results, frame)
        
        for i, text in enumerate([f"Emotion: {emotion}", "Hand on Face" if hand_on_face else "", head_pose_text, eye_text]):
            if text:
                cv2.putText(frame, text, (20, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, f'FPS: {int(1 / (time.time() - start))}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 176, 242), 2)
        cv2.imshow('AI Multimodal Analysis', frame)
        
        frame_numbers.append(frame_count)
        hand_on_face_history.append(hand_on_face)
        eye_text_history.append(eye_text)
        head_pose_text_history.append(head_pose_text)
        emotion_history.append(emotion)
        seconds.append(int(time.time() - second))

        # Update hand, eye, head, emoji counters
        time_ = alarm(delay, time_, hand_on_face, head_pose_text, eye_text, emotion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    return {
        "frame_numbers": frame_numbers,
        "hand_on_face": hand_on_face_history,
        "eye_text": eye_text_history,
        "head_pose": head_pose_text_history,
        "emotion": emotion_history,
        "seconds": seconds
    }

if __name__ == "__main__":
    results = main()
    df = pd.DataFrame(results)

def feedback(df):   
    import numpy as np
    import pandas as pd 
    import time
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go

    # Convert categorical labels to numeric for plotting
    df['seconds'] = df['seconds'].astype(int)
    df['hand_on_face'] = df['hand_on_face'].astype(int)
    df['head_pose_numeric'] = df['head_pose'].apply(lambda x: 1 if x == 'Looking Forward' else 0)
    df['eye_numeric'] = df['eye_text'].apply(lambda x: 1 if x == 'Eye Center' else 0)
    df['emotion_numeric'] = df['emotion'].apply(lambda x: 1 if x in ['Neutral', 'Happiness'] else 0)

    # Helper: convert seconds to minute:second format for x-axis ticks
    df['minute_str'] = df['seconds'].apply(lambda x: f"{x//60}:{x%60:02d}")

    df_grouped = df.groupby('seconds')[['hand_on_face', 'head_pose_numeric', 'eye_numeric', 'emotion_numeric']].agg({
        'hand_on_face': 'min',
        'head_pose_numeric': 'max',
        'eye_numeric': 'max',
        'emotion_numeric': 'max'
    }).reset_index()

    df_grouped['group_hand'] = (df_grouped['hand_on_face'] != df_grouped['hand_on_face'].shift()).cumsum()
    df_grouped['group_head'] = (df_grouped['head_pose_numeric'] != df_grouped['head_pose_numeric'].shift()).cumsum()
    df_grouped['group_eye'] = (df_grouped['eye_numeric'] != df_grouped['eye_numeric'].shift()).cumsum()
    df_grouped['group_emotion'] = (df_grouped['emotion_numeric'] != df_grouped['emotion_numeric'].shift()).cumsum()

    group_hand = df_grouped.groupby(['group_hand', 'hand_on_face']).size().reset_index(name='count_hand')
    df_grouped = df_grouped.merge(group_hand, on=['group_hand','hand_on_face'])

    group_head = df_grouped.groupby(['group_head', 'head_pose_numeric']).size().reset_index(name='count_head')
    df_grouped = df_grouped.merge(group_head, on=['group_head','head_pose_numeric'])

    group_eye = df_grouped.groupby(['group_eye', 'eye_numeric']).size().reset_index(name='count_eye')
    df_grouped = df_grouped.merge(group_eye, on=['group_eye','eye_numeric'])

    group_emotion = df_grouped.groupby(['group_emotion', 'emotion_numeric']).size().reset_index(name='count_emotion')
    df_grouped = df_grouped.merge(group_emotion, on=['group_emotion','emotion_numeric'])

    df_grouped['hand_summary'] = df_grouped.apply(
        lambda row: 0 if row['hand_on_face'] == 1 and row['count_hand'] >= 30 else 1,
        axis=1)

    df_grouped['head_pose_summary'] = df_grouped.apply(
        lambda row: 0 if row['head_pose_numeric'] == 0 and row['count_head'] >= 30 else 1,
        axis=1)

    df_grouped['eye_summary'] = df_grouped.apply(
        lambda row: 0 if row['eye_numeric'] == 0 and row['count_eye'] >= 30 else 1,
        axis=1)


    df_grouped['emotion_summary'] = df_grouped.apply(
        lambda row: 0 if row['emotion_numeric'] == 0 and row['count_emotion'] >= 30 else 1,
        axis=1)

    summary = ['head_pose_summary', 'eye_summary', 'hand_summary', 'emotion_summary']
    df_grouped['final_summary'] = df_grouped[summary].min(axis=1)
    df_grouped['lowest_source'] = df_grouped[summary].apply(
        lambda row: ', '.join([col for col in summary if row[col] == 0]) if row.min() == 0 else '',
        axis=1
    )

    fig = go.Figure()
    df_grouped['minute_str'] = df_grouped['seconds'].apply(lambda x: f"{x//60}:{x%60:02d}")

    xticks = df_grouped['seconds'][::len(df_grouped)//10]  
    xtick_labels = df_grouped['minute_str'][::len(df_grouped)//10]

    fig.add_trace(go.Scatter(x=df_grouped['seconds'], y=df_grouped['hand_summary'],
                            mode='lines', name='Nervous', line=dict(color='red')))

    df_grouped['distraction_summary'] = np.minimum(
        df_grouped['head_pose_summary'],
        df_grouped['eye_summary']
    )

    fig.add_trace(go.Scatter(
        x=df_grouped['seconds'],
        y=df_grouped['distraction_summary'],
        mode='lines',
        name='Distraction',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(x=df_grouped['seconds'], y=df_grouped['emotion_summary'],
                            mode='lines', name='Distruct', line=dict(color='green')))

    fig.update_layout(title="Summary of Interview",
                    xaxis_title="Time (mm:ss)",
                    yaxis_title="Focus in Intreview",
                    yaxis=dict(tickvals=[0, 1], ticktext=['No', 'Yes']),
                    xaxis=dict(tickmode='array', tickvals=xticks, ticktext=xtick_labels))

    # fig.show()

    df_grouped['summary_change_group'] = (df_grouped['lowest_source'] != df_grouped['lowest_source'].shift()).cumsum()

    summary_ranges = df_grouped.groupby(['summary_change_group', 'lowest_source'])['seconds'].agg(['min', 'max']).reset_index()

    summary_ranges = summary_ranges[summary_ranges['lowest_source'] != '']

    summary_ranges['duration'] = summary_ranges['max'] - summary_ranges['min']
    summary_ranges = summary_ranges[summary_ranges['duration'] >= 29]

    summary_mapping = {
        'hand_summary': 'you put Hand on Face',
        'head_pose_summary': 'you moved your Head alot',
        'eye_summary': 'you moved your Eye alot',
        'emotion_summary': 'you change in facial expressions'
    }
    summary_analysis = {
        'hand_summary': 'you were Nervous',
        'head_pose_summary': 'you were Distractded',
        'eye_summary': 'you were Distracted',
        'emotion_summary': 'you have no confidence'
    }

    summary_text = []
    print('Head Pose Estimation ---> Distraction')
    print('Eye Tracking ---> Distraction')
    print('Hand on Face ---> Nervous')
    print('Emotional ---> Distruct')
    print('------------------------------')
    for index, row in summary_ranges.iterrows():
        start = row['min']
        end = row['max']
        source_list = row['lowest_source'].split(', ')

        for source in source_list:
            if source in summary_mapping:
                start_min = str(start // 60)
                start_sec = str(start % 60).zfill(2)
                end_min = str(end // 60)
                end_sec = str(end % 60).zfill(2)
                text = f"From {start_min}:{start_sec} to {end_min}:{end_sec} Minutes, {summary_analysis[source]} because {summary_mapping[source]}"
                summary_text.append(text)
    return fig, summary_text   

if __name__ == "__main__":
    fig, summary_text = feedback(df)
    for txt in summary_text:
        print(txt)
    fig.show()