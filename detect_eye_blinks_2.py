"""状態遷移マトリックスを使用した、まばたきの検知

状態遷移を表す2x2のマトリックスを作成します。このマトリックスは、前の状態と現在の状態に基づいて次の状態を決定します。

| 前の状態 | 現在の状態 | 次の状態 |
|----------|------------|----------|
| OPEN     | OPEN       | OPEN     |
| OPEN     | CLOSED     | CLOSED   |
| CLOSED   | OPEN       | BLINKING |
| CLOSED   | CLOSED     | CLOSED   |

"""

import sys

sys.path.append('/usr/lib/python3/dist-packages')
import time
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np

# 顔のランドマークを検出するためのモデルをロード
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ビデオファイルからの入力を開始
cap = cv2.VideoCapture('assets/happy.mp4')
# cap = cv2.VideoCapture('assets/man.mp4')

# 目の状態を表す列挙型
class EyeState(Enum):
    OPEN = 1
    CLOSED = 2
    BLINKING = 3

current_state = EyeState.OPEN
previous_state = EyeState.OPEN

# 最初のフレームを処理したかどうかを追跡するフラグ
first_frame_processed = False

# EARの閾値
EAR_THRESHOLD_CLOSE = 1.4
EAR_THRESHOLD_OPEN = 1.2

blink_count = 0  # 瞬きの回数をカウント

frame_time = 1 / 30  # 1フレームの時間（秒単位）. 30fpsの場合は1/30

def calculate_eye_ratio(face_landmarks, eye_landmarks):
    # 眼のアスペクト比を計算する関数
    eye_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in eye_landmarks])
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_ratio = (A + B) / (2.0 * C)
    return eye_ratio

while cap.isOpened():
    start_time = time.time()
    ret, image = cap.read()
    if not ret:
        print("画像を取得できませんでした")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_ratio = calculate_eye_ratio(face_landmarks, [33, 246, 161, 160, 159, 158, 157, 173])
            right_eye_ratio = calculate_eye_ratio(face_landmarks, [263, 466, 388, 387, 386, 385, 384, 398])

        # 前回の状態を保存
        previous_state = current_state

        # 状態遷移マトリックスに基づいて目の状態を更新
        if left_eye_ratio < EAR_THRESHOLD_CLOSE or right_eye_ratio < EAR_THRESHOLD_CLOSE:
            current_state = EyeState.CLOSED
        elif left_eye_ratio > EAR_THRESHOLD_OPEN or right_eye_ratio > EAR_THRESHOLD_OPEN:
            current_state = EyeState.OPEN

        # 最初のフレームの場合、瞬きのカウントをスキップ
        if not first_frame_processed:
            first_frame_processed = True
            continue

        # 前回の状態がOPEN、現在の状態がCLOSED、次の状態が再びOPENとなる場合に瞬きとしてカウント
        if previous_state == EyeState.OPEN and current_state == EyeState.CLOSED:
            blink_count += 1

        cv2.putText(image, f"Blink: {blink_count}", (image.shape[1] - 200, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('MediaPipe FaceMesh', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # 1フレームの処理が終わったら、次のフレームまでの待機時間を計算して待機
    elapsed_time = time.time() - start_time
    sleep_time = frame_time - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
