
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/bird_and_girl.png)

## はじめに
皆さんは、鳥の「頭を静止させる不思議な能力」をご存知ですか？
試しに「bird head stability」などで検索をかけてみてください。とてもかわいいですよ。
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/bird_head_stabilzation.gif)

調べてみましたが、この現象に特定の名前があるわけではないようです。
この能力は鳥の首関節の驚異的な柔軟性に起因するそうで、頭を静止させることで、激しい飛行中の環境内オブジェクト追跡を可能にしているそうです。

この性質を利用して、鳥の頭にカメラを取り付け、スタビライザー（ジンバル）のように動かないカメラを作る猛者もいるようです。アホですね（褒め言葉）。大好きです。

![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/2023-08-29-19-24-43.png)

![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/2023-08-29-19-26-01.png)

https://www.youtube.com/watch?v=8A5cMcsYVHY

さて、この鳥の頭のように、人間の頭も相対的に静止させることができるでしょうか？
画像処理の技術を使えば、人間の頭を静止させることができます。

## 元動画
まずは、元動画をご覧ください。
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/woman_dance.gif)

動画内で顔の位置が動く場合に、その顔を一定の位置に固定します。
具体的には、顔の中心を両目の中心に設定し、その位置を固定します。

https://github.com/yKesamaru/FacePositionStabilization/blob/57a2c6bb66d3ab325a40629872088c6bc9cb7d36/face_position.py#L31-L69


## 各部分の詳細
コードの各行を順番に見ていきましょう。

### 顔の検出
```python
results = face_mesh.process(image_rgb)
```

RGB形式の画像（`image_rgb`）をMediaPipeのFace Meshモデルに渡して、顔のランドマークを検出します。

### 両目の中心を計算

```python
left_eye = face_landmarks.landmark[33]
right_eye = face_landmarks.landmark[263]
eye_x = int((left_eye.x + right_eye.x) * width // 2)
eye_y = int((left_eye.y + right_eye.y) * height // 2)
```

ここでは、検出された顔のランドマークから左目（`landmark[33]`）と右目（`landmark[263]`）の位置を取得し、それらの平均位置を画像内での座標（`eye_x`, `eye_y`）に変換します。

### 両目の距離とスケーリングファクターの計算

```python
eye_distance = int(np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) * width)
scale_factor = fixed_eye_distance / eye_distance
```

この部分で、両目の距離（`eye_distance`）とスケーリングファクター（`scale_factor`）を計算します。スケーリングファクターは、固定したい両目の距離（`fixed_eye_distance`）を実際の両目の距離で割って求めます。

### 平行移動とスケーリング

```python
M_translate_scale = np.float32([[scale_factor, 0, fixed_eye_x - eye_x * scale_factor],
                                [0, scale_factor, fixed_eye_y - eye_y * scale_factor]])
image = cv2.warpAffine(image, M_translate_scale, (width, height))
```

`cv2.warpAffine`関数を使用して、計算したスケーリングファクターと両目の位置に基づいて画像を平行移動とスケーリングを行います。

### 画像の回転

```python
angle = np.arctan2((left_eye.y - right_eye.y) * height, (left_eye.x - right_eye.x) * width)
angle = np.degrees(angle)
M_rotate = cv2.getRotationMatrix2D((fixed_eye_x, fixed_eye_y), angle, 1)
image = cv2.warpAffine(image, M_rotate, (width, height))
```

最後に`cv2.warpAffine`を再度使用して、画像を回転させます。回転角度は、両目の位置に基づいて`np.arctan2`関数で計算します。



## 結果
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/woman_head_stabilization.gif)
できました！
顔の位置が固定されていますね。

## まとめ
顔の位置を固定する技術は、多くの応用分野で活用されます。

### 顔認証・顔検出
顔の位置が固定されていると、顔認証や顔検出の精度が向上します。顔が動画内で頻繁に動く場合や、角度が変わる場合に有用です。

### ビデオ会議
ビデオ会議での顔の位置を固定することで、参加者が話をする際に顔がしっかりとフレーム内に収まるようになります。

### ヒューマンコンピュータインタラクション（HCI）
顔の動きをトラッキングすることで、より直感的なインターフェイスやコントロールが可能になります。たとえば顔の位置に応じてカーソルを動かすといった応用が考えられます。

### 医療診断
顔の特定の部分（たとえば、目や口）に焦点を当てる必要がある医療診断でも、この技術は有用です。

以上です。
今回は鳥の頭のように、人間の頭も相対的に静止させる方法を紹介しました。

