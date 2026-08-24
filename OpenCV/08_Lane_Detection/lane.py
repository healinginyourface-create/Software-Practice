import cv2
import os
import numpy as np


# 프로젝트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

video_path = os.path.join(
    BASE_DIR,
    "assets",
    "drivingvideo.mp4"
)


# 영상 불러오기
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("영상을 찾을 수 없습니다.")
    exit()


while True:

    # 프레임 읽기
    ret, frame = video.read()

    if not ret:
        break


    # ============================
    # 1. Grayscale
    # ============================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ============================
    # 2. Canny Edge
    # ============================

    edges = cv2.Canny(
        gray,
        50,
        150
    )


    # ============================
    # 3. ROI
    # ============================

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

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    roi = cv2.bitwise_and(
        edges,
        mask
    )


    # ============================
    # 4. Hough Transform
    # ============================

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=20
    )


    # ============================
    # 5. 차선 표시
    # ============================

    result = frame.copy()

    if lines is not None:

        for line in lines:

            line = np.asarray(line).reshape(-1)

            if len(line) != 4:
                continue

            x1, y1, x2, y2 = line

            # 수직선 방지
            if x2 == x1:
                continue

            # 기울기
            slope = (y2 - y1) / (x2 - x1)

            # 너무 수평인 선 제거
            if abs(slope) < 0.5:
                continue

            cv2.line(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                3
            )


    # ============================
    # 결과 출력
    # ============================

    cv2.imshow(
        "Lane Detection",
        result
    )


    # Q를 누르면 종료
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()


print("차선 검출 종료")