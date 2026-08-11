import cv2
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "assets", "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 1. 흑백 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 노이즈 제거
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Edge 검출
edges = cv2.Canny(blur, 50, 150)

plt.imshow(edges, cmap="gray")
plt.title("Canny Edge")
plt.axis("off")
plt.show()