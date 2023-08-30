
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/blink.png)

- [はじめに](#はじめに)
- [まばたき検知の原理](#まばたき検知の原理)
- [手順](#手順)
  - [入力動画](#入力動画)
  - [コード](#コード)
  - [出力動画](#出力動画)
- [まとめ](#まとめ)


## はじめに
顔のなりすまし対策は、顔認識システムのセキュリティにとって重要です。
まばたき検知は、このアンチスプーフィング手法の1つとして用いられます。

アンチスプーフィングについてのサーベイ論文は、2023年の[Deep Learning for Face Anti-Spoofing: A Survey](https://arxiv.org/abs/2106.14948)が詳しいです。この論文ではさまざまなアンチスプーフィング手法と、それに対する研究が紹介されています。

![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/2023-08-30-18-44-19.png)
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/2023-08-30-19-08-11.png)

> Most traditional algorithms are designed based on human liveness cues and handcrafted features, which need rich task-aware prior knowledge for design. In term of the methods based on the liveness cues, eye-blinking [2], [7], [8], face and head movement [9], [10] (e.g., nodding and smiling), gaze tracking [11], [12] and remote physiological signals (e.g., rPPG [3], [13], [14], [15]) are explored for dynamic discrimination.
> However, these physiological liveness cues are usually captured from long-term interactive face videos, which is inconvenient for practical deployment.
> 
> 従来のアルゴリズムのほとんどは、生体であるかどうかこちらから問いかけをする方法とマニュアルの特徴量に基づいて設計されており、設計にはタスクを意識した豊富な事前知識が必要です。生体情報に基づく方法としては、まばたき [2]、[7]、[8]、顔や頭の動き [9]、[10] (うなずきや笑顔など)、視線追跡 [11] 、[12] および遠隔生理学的信号 (例: rPPG [3]、[13]、[14]、[15]) は動的識別のために研究されています。
> ただし、これらの生体認証は通常、長期にわたるインタラクティブな顔のビデオからキャプチャされるため、実際の展開には不便です。

このような理由から、近年ではディープラーニングを用いたアンチスプーフィング手法が注目されています。

とはいえ、ディープラーニングを用いたアンチスプーフィング手法は、データセットの不足や、データセットの偏り、データセットの大きさなどの問題があります。また、照度やカメラ性能により、現実のシーンでは使いづらい面が多々あります。そのため、ディープラーニングを用いたアンチスプーフィング手法は、まだまだ研究段階にあります。

この記事では、伝統的なアンチスプーフィング手法の1つである、まばたき検知をPythonで実装してみます。

## まばたき検知の原理
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/2023-08-30-19-53-26.png)
まばたきに関する論文は、2016年の[Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)に掲載されています。

眼のアスペクト比（Eye Aspect Ratio: EAR）は、目の開き具合を数値で表す指標です。EARは目のランドマーク（特定の点）に基づいて計算されます。具体的には、目の上縁と下縁にある点$( p_2, p_3, p_5, p_6 )$と、目の左端と右端にある点$( p_1, p_4 )$を使用します。

$$\text{EAR} = \frac{{||p_2 - p_6|| + ||p_3 - p_5||}}{2 \times ||p_1 - p_4||}$$

この式で使用されている $( || \cdot || )$ はユークリッド距離を表します。

- $( ||p_2 - p_6|| )$ と $( ||p_3 - p_5|| )$ は、それぞれ目の上縁と下縁の距離を計算します。これらの距離が大きいほど、目は開いていると言えます。
- $( ||p_1 - p_4|| )$ は目の左端と右端の距離を計算します。この距離は目が開いているか閉じているかにかかわらず、ほぼ一定です。

EARの値は、目が開いているときには比較的大きく、目が閉じているときには小さくなります。この性質を利用して、瞬きを検出できます。具体的には、EARがある閾値よりも小さくなったときに瞬きが発生したと判断することが一般的です。

MediaPipeのFace Meshモデルにおいて、目のランドマークは以下のように対応しています：

- 左目（左から右へ）
  - $( p_1 )$ : 33
  - $( p_2 )$ : 159
  - $( p_3 )$ : 145
  - $( p_4 )$ : 133
  - $( p_5 )$ : 153
  - $( p_6 )$ : 144

- 右目（左から右へ）
  - $( p_1 )$ : 263
  - $( p_2 )$ : 386
  - $( p_3 )$ : 374
  - $( p_4 )$ : 362
  - $( p_5 )$ : 380
  - $( p_6 )$ : 373


![](https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png)

くわしくは、[こちらの公式ドキュメント](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)を参照してください。

これらのランドマーク番号を使用して、EARを計算できます。この情報を元に、`calculate_eye_ratio`関数を適切に修正することで、EARに基づいた瞬き検出が可能です。

実際は、トライアンドエラーを繰り返して「EARの閾値」を決定します。この閾値は、環境によって異なるため、一般的な値は存在しません。

## 手順
### 入力動画
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/happy.gif)
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/man.gif)
### コード
まばたき検知のコードは、以下のようになります。

https://github.com/yKesamaru/FacePositionStabilization/blob/dcfa2203dc820d62684a5d159d58e3b229ff866d/detect_eye_blinks.py#L1-L96

EARの閾値は以下のように設定しました。
https://github.com/yKesamaru/FacePositionStabilization/blob/dec28210c7ca2921650562e25cade8e6a33c8c21/detect_eye_blinks.py#L21-L23

EARを求める関数を以下のように定義します。
https://github.com/yKesamaru/FacePositionStabilization/blob/dcfa2203dc820d62684a5d159d58e3b229ff866d/detect_eye_blinks.py#L29-L37

まばたき検知は

- 目が開いている
- 目が閉じている

という動作を繰り返します。この動作を表現するために、以下のようにコーディングしました。
https://github.com/yKesamaru/FacePositionStabilization/blob/dcfa2203dc820d62684a5d159d58e3b229ff866d/detect_eye_blinks.py#L65-L72

### 出力動画
表示画面の右下に、瞬きの回数が表示されます。

![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/happy_output.gif)
![](https://raw.githubusercontent.com/yKesamaru/FacePositionStabilization/master/assets/man_output.gif)

## まとめ
この記事では伝統的なアンチスプーフィング技術である「まばたき検知」を解説し、実際のコードを紹介しました。

以上です。ありがとうございました。

