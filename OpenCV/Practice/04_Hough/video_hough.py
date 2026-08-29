import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "assets", "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

output_dir = os.path.join(BASE_DIR, "hough_frames")
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while frame_count < 10:

    frame = image.copy()
    frame_count += 1

    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Canny
    edges = cv2.Canny(gray, 50, 150)

    # 3. Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )

    # 4. 검출된 직선 그리기
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # 5. 결과 저장
    output_path = os.path.join(
        output_dir,
        f"frame_{frame_count:02d}.jpg"
    )

    cv2.imwrite(output_path, frame)

    print(f"Frame {frame_count} 처리 완료")

print("Hough 영상 처리 완료")
print(f"결과 위치: {output_dir}")