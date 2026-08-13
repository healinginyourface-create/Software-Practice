import cv2
import os

# 현재 파일 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenCV 폴더
OPENCV_DIR = os.path.dirname(CURRENT_DIR)

# 이미지 경로
image_path = os.path.join(
    OPENCV_DIR,
    "assets",
    "car.jpg"
)

# 이미지 읽기
image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    print(f"확인한 경로: {image_path}")
    exit()

print("이미지 불러오기 성공")
print(f"이미지 크기: {image.shape}")


# 1. Grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# 2. Threshold
_, binary = cv2.threshold(
    gray,
    120,
    255,
    cv2.THRESH_BINARY
)


# 3. Contour 검출
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"검출된 Contour: {len(contours)}")


# 4. 큰 Contour만 Bounding Box 표시
object_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # 너무 작은 영역 제거
    if area < 1000:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Bounding Box
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    object_count += 1

    print(
        f"객체 {object_count}: "
        f"위치=({x}, {y}), "
        f"크기=({w}, {h}), "
        f"면적={area:.1f}"
    )


# 5. 결과 저장
output_path = os.path.join(
    CURRENT_DIR,
    "detected.jpg"
)

cv2.imwrite(
    output_path,
    image
)

print()
print(f"검출된 객체 수: {object_count}")
print(f"결과 저장 위치: {output_path}")