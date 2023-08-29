import sys

sys.path.append('/usr/lib/python3/dist-packages')

import cv2
import numpy as np

# 黒い画像を生成
img = np.zeros((512, 512, 3), np.uint8)

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
