import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("ding.wav")  # Replace with your sound file path

cap = cv2.VideoCapture(0)
rep_count = 0
stage = None
last_play_time = 0
cooldown = 1.0  # seconds

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < 50:
                stage = "up"
            if angle > 160 and stage == "up":
                stage = "down"
                rep_count += 1
                current_time = time.time()
                if current_time - last_play_time > cooldown:
                    sound.play()
                    last_play_time = current_time

            cv2.putText(image, f'REPS: {rep_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'STAGE: {stage}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Shoulder Press Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
