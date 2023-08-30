import sys

sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import mediapipe as mp
import numpy as np
import time

# 顔のランドマークを検出するためのモデルをロード
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# ビデオファイルからの入力を開始
cap = cv2.VideoCapture('assets/man.mp4')
# cap = cv2.VideoCapture('assets/happy.mp4')

# 眼の状態を追跡する変数
eye_open = True

# EARの閾値
EAR_THRESHOLD_CLOSE = 1.4  # 目が閉じていると判断する閾値
EAR_THRESHOLD_OPEN = 1.2   # 目が開いていると判断する閾値

blink_count = 0  # 瞬きの回数をカウント

frame_time = 1 / 30  # 1フレームの時間（秒単位）. 30fpsの場合は1/30

def calculate_eye_ratio(face_landmarks, eye_landmarks):
    # 眼のアスペクト比を計算する関数
    eye_points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] for i in eye_landmarks])
    # EAR計算
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    eye_ratio = (A + B) / (2.0 * C)
    return eye_ratio

# 動画のフレームサイズを取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))

while cap.isOpened():
    start_time = time.time()  # フレーム処理の開始時間
    ret, image = cap.read()  # ビデオから画像を読み取る
    if not ret:
        print("画像を取得できませんでした")
        break

    # 画像を処理するためにBGRからRGBに変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 顔のランドマークを検出
    results = face_mesh.process(image_rgb)

    # 瞬きを検出
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_ratio = calculate_eye_ratio(face_landmarks, [33, 246, 161, 160, 159, 158, 157, 173])
            right_eye_ratio = calculate_eye_ratio(face_landmarks, [263, 466, 388, 387, 386, 385, 384, 398])

            # 目が閉じていると判断
            if left_eye_ratio < EAR_THRESHOLD_CLOSE or right_eye_ratio < EAR_THRESHOLD_CLOSE:
                eye_open = False
            # 目が開いていると判断
            elif left_eye_ratio > EAR_THRESHOLD_OPEN or right_eye_ratio > EAR_THRESHOLD_OPEN:
                if not eye_open:
                    blink_count += 1  # 瞬きの回数を増やす
                eye_open = True

        # 瞬きの回数を画面の右下に描画（フォントサイズは30pt）
        cv2.putText(image, f"Blink: {blink_count}", (image.shape[1] - 200, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 画像を表示
    cv2.imshow('Blink', image)
    out.write(image)

    # 'q'キーで終了
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    end_time = time.time()  # フレーム処理の終了時間
    elapsed_time = end_time - start_time  # フレーム処理にかかった時間（秒）

    # 次のフレームまでの待機時間を計算
    wait_time = max(0, frame_time - elapsed_time)

    # 指定した時間だけ待機
    time.sleep(wait_time)

# ビデオを解放
cap.release()
cv2.destroyAllWindows()

