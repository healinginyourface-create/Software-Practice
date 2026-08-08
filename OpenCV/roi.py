import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Canny Edge
edges = cv2.Canny(blur, 50, 150)

# 4. ROI 마스크 생성
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (width * 0.55, height * 0.55),
    (width * 0.45, height * 0.55)
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)

# 5. ROI 적용
roi = cv2.bitwise_and(edges, mask)

# 6. 결과 출력
plt.imshow(roi, cmap="gray")
plt.title("ROI + Canny")
plt.axis("off")
plt.show()