import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# 이미지 경로
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

# 4. ROI
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([[
    (0, height),
    (width, height),
    (int(width * 0.60), int(height * 0.55)),
    (int(width * 0.40), int(height * 0.55))
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)

roi = cv2.bitwise_and(edges, mask)

# 5. Hough Lines
lines = cv2.HoughLinesP(
    roi,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=50
)

# 6. 결과 이미지
result = image.copy()

if lines is not None:

    for line in lines:

        x1, y1, x2, y2 = line[0]

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

# 7. 출력
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))
plt.imshow(result)
plt.title("Hough Lines")
plt.axis("off")
plt.show()