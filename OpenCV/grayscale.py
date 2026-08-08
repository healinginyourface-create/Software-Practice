import cv2
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 컬러 → 흑백
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("원본 크기:", image.shape)
print("흑백 크기:", gray.shape)

plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")
plt.show()