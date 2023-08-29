import sys

sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのFaceMeshモデルを初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# input.mp4から映像を取得
cap = cv2.VideoCapture("assets/input.mp4")

# 動画のフレームサイズを取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 両目の固定位置（画像の中心に設定）
fixed_eye_x = width // 2
fixed_eye_y = height // 2

# 両目の固定距離（ピクセル単位で設定、例えば100）
fixed_eye_distance = 100

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # BGRからRGBに変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 顔の検出
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 両目の中心を計算
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            eye_x = int((left_eye.x + right_eye.x) * width // 2)
            eye_y = int((left_eye.y + right_eye.y) * height // 2)

            # 両目の距離を計算
            eye_distance = int(np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) * width)

            # スケーリングファクターを計算
            scale_factor = fixed_eye_distance / eye_distance

            # 平行移動とスケーリングを行う
            M_translate_scale = np.float32([[scale_factor, 0, fixed_eye_x - eye_x * scale_factor],
                                            [0, scale_factor, fixed_eye_y - eye_y * scale_factor]])
            image = cv2.warpAffine(image, M_translate_scale, (width, height))

            # 両目の位置に基づいて回転角度を計算
            angle = np.arctan2((left_eye.y - right_eye.y) * height, (left_eye.x - right_eye.x) * width)
            angle = np.degrees(angle)

            # 回転行列を計算
            M_rotate = cv2.getRotationMatrix2D((fixed_eye_x, fixed_eye_y), angle, 1)

            # 画像を回転
            image = cv2.warpAffine(image, M_rotate, (width, height))

            # 画像を上下反転
            image = cv2.flip(image, 0)

    # 出力動画にフレームを追加
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    out.write(image)
    
    # imshow()で画面に表示
    cv2.imshow('test', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()
