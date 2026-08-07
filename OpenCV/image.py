import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 일부 자르기
crop = image_rgb[300:900, 200:900]

plt.imshow(crop)
plt.title("Crop")
plt.axis("off")
plt.show()