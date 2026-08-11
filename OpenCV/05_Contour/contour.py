import cv2
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "assets", "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Threshold
_, binary = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_BINARY
)

# 3. Contour
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

count = 0

# 4. Contour 분석
for contour in contours:

    area = cv2.contourArea(contour)

    # 작은 영역 제거
    if area < 150:
        continue

    count += 1

    # Bounding Box
    x, y, w, h = cv2.boundingRect(contour)

    # Moments
    M = cv2.moments(contour)

    # 중심점 계산
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = 0
        cy = 0

    # Contour
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Bounding Box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # 중심점
    cv2.circle(
        result,
        (cx, cy),
        6,
        (0, 0, 255),
        -1
    )

    # 중심점 좌표 표시
    cv2.putText(
        result,
        f"({cx}, {cy})",
        (cx + 10, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    print(f"Contour {count}")
    print(f"  면적: {area:.1f}")
    print(f"  Bounding Box: ({x}, {y}, {w}, {h})")
    print(f"  중심점: ({cx}, {cy})")
    print()

# 5. 출력
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(result)
plt.title("Contour + Centroid")
plt.axis("off")
plt.show()

print("검출된 Contour 개수:", count)