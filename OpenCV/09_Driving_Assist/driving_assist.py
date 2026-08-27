import cv2
import os
import numpy as np
from ultralytics import YOLO


# ============================
# 프로젝트 경로
# ============================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OPENCV_DIR = os.path.dirname(
    CURRENT_DIR
)


# ============================
# 영상 경로
# ============================

video_path = os.path.join(
    OPENCV_DIR,
    "assets",
    "drivingvideo.mp4"
)


# ============================
# YOLO 모델
# ============================

model_path = os.path.join(
    OPENCV_DIR,
    "yolo11n.pt"
)

model = YOLO(model_path)


# ============================
# 차량 클래스
# ============================

# car = 2
# motorcycle = 3
# bus = 5
# truck = 7

vehicle_classes = [2, 3, 5, 7]


# ============================
# 영상 불러오기
# ============================

video = cv2.VideoCapture(
    video_path
)

if not video.isOpened():

    print("영상을 찾을 수 없습니다.")
    print(f"확인한 경로: {video_path}")

    exit()


print("영상 불러오기 성공")
print("Driving Assist 시작")


# ============================
# 영상 처리
# ============================

while True:

    ret, frame = video.read()

    if not ret:
        break


    # ==================================================
    # 1. LANE DETECTION
    # ==================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Canny Edge

    edges = cv2.Canny(
        gray,
        50,
        150
    )


    # ROI

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([
        [
            (0, height),
            (width, height),
            (
                int(width * 0.6),
                int(height * 0.55)
            ),
            (
                int(width * 0.4),
                int(height * 0.55)
            )
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


    # Hough Transform

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=20
    )


    # ==================================================
    # 2. 결과 프레임
    # ==================================================

    result = frame.copy()


    # 차선 표시

    if lines is not None:

        for line in lines:

            line = np.asarray(
                line
            ).reshape(-1)


            if len(line) != 4:
                continue


            x1, y1, x2, y2 = line


            # 수직선 제거

            if x2 == x1:
                continue


            # 기울기

            slope = (
                (y2 - y1)
                /
                (x2 - x1)
            )


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


    # ==================================================
    # 3. YOLO OBJECT DETECTION
    # ==================================================

    results = model(
        frame,
        verbose=False
    )

    detection = results[0]


    vehicle_count = 0


    if detection.boxes is not None:

        for box in detection.boxes:

            class_id = int(
                box.cls[0]
            )


            # 차량만 검출

            if class_id not in vehicle_classes:
                continue


            vehicle_count += 1


            # Bounding Box

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Confidence

            confidence = float(
                box.conf[0]
            )


            # 클래스 이름

            class_name = (
                detection.names[class_id]
            )


            # 차량 박스

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )


            # Label

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                result,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


    # ==================================================
    # 4. 차량 개수
    # ==================================================

    cv2.putText(
        result,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # ==================================================
    # 5. 화면 출력
    # ==================================================

    cv2.imshow(
        "Driving Assist",
        result
    )


    # 기존 영상 속도에 맞춤

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# ============================
# 종료
# ============================

video.release()

cv2.destroyAllWindows()

print("Driving Assist 종료")

# python3 09_Driving_Assist/driving_assist.py