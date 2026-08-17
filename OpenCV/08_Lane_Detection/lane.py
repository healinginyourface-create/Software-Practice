import cv2
import os
import numpy as np


# 프로젝트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "assets", "road.jpg")

# 이미지 불러오기
image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()


# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2. Canny Edge
edges = cv2.Canny(gray, 50, 150)


# 3. ROI 설정
height, width = edges.shape

mask = np.zeros_like(edges)

polygon = np.array([
    [
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.55)),
        (int(width * 0.4), int(height * 0.55))
    ]
], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)

roi = cv2.bitwise_and(edges, mask)


# 4. Hough Transform
lines = cv2.HoughLinesP(
    roi,
    1,
    np.pi / 180,
    threshold=30,
    minLineLength=40,
    maxLineGap=20
)


# 5. 결과 이미지
result = image.copy()

if lines is not None:

    for line in lines:

        # OpenCV 버전에 따라 반환 형태가 달라도 처리
        line = np.asarray(line).reshape(-1)

        if len(line) != 4:
            continue

        x1, y1, x2, y2 = line

        # 수직선 방지
        if x2 == x1:
            continue

        # 기울기 계산
        slope = (y2 - y1) / (x2 - x1)

        # 너무 수평인 선 제거
        if abs(slope) < 0.5:
            continue

        # 차선 후보 표시
        cv2.line(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )


# 6. 결과 저장
output_path = os.path.join(
    os.path.dirname(__file__),
    "lane_result.jpg"
)

cv2.imwrite(output_path, result)


print("차선 검출 완료")
print(f"결과 파일: {output_path}")