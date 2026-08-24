import cv2
import os
from ultralytics import YOLO


# 현재 파일 위치
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# OpenCV 폴더
OPENCV_DIR = os.path.dirname(CURRENT_DIR)

# 영상 경로
video_path = os.path.join(
    OPENCV_DIR,
    "assets",
    "drivingvideo.mp4"
)


# YOLO 모델
model = YOLO("yolo11n.pt")


# 영상 불러오기
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("영상을 찾을 수 없습니다.")
    print(f"확인한 경로: {video_path}")
    exit()

print("영상 불러오기 성공")


# COCO 클래스
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7

vehicle_classes = [2, 3, 5, 7]


while True:

    ret, frame = video.read()

    if not ret:
        break


    # YOLO 검출
    results = model(
        frame,
        verbose=False
    )

    result = results[0]


    # 차량 개수
    vehicle_count = 0


    # 검출된 객체 확인
    if result.boxes is not None:

        for box in result.boxes:

            # 클래스 번호
            class_id = int(
                box.cls[0]
            )

            # 차량인지 확인
            if class_id not in vehicle_classes:
                continue


            # 차량 개수 증가
            vehicle_count += 1


            # Bounding Box 좌표
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # 신뢰도
            confidence = float(
                box.conf[0]
            )


            # 클래스 이름
            class_name = result.names[class_id]


            # Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # 이름 + confidence
            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # 차량 수 표시
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # 결과 출력
    cv2.imshow(
        "Vehicle Detection",
        frame
    )


    # Q 종료
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

print("Vehicle Detection 종료")