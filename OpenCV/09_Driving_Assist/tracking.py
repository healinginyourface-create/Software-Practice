import cv2
import os
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
# 영상 / 모델
# ============================

video_path = os.path.join(
    OPENCV_DIR,
    "assets",
    "drivingvideo.mp4"
)

model_path = os.path.join(
    OPENCV_DIR,
    "yolo11n.pt"
)

model = YOLO(model_path)


# ============================
# 영상
# ============================

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("영상을 찾을 수 없습니다.")
    exit()

print("영상 불러오기 성공")
print("Tracking + Distance Detection 시작")


# ============================
# 설정
# ============================

# 화면 중앙 기준
LANE_CENTER = 0.5

# 차량이 화면의 어느 정도 아래쪽에 있으면
# 앞차 후보로 판단
FRONT_Y_RATIO = 0.45


# ============================
# 메인 루프
# ============================

while True:

    ret, frame = video.read()

    if not ret:
        break


    height, width = frame.shape[:2]


    # ============================
    # YOLO Tracking
    # ============================

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        classes=[2, 3, 5, 7]
    )

    result = results[0]


    # 앞차 후보
    front_vehicle = None


    # ============================
    # 차량 처리
    # ============================

    if result.boxes is not None:

        for box in result.boxes:

            if box.id is None:
                continue


            # ----------------------------
            # Tracking ID
            # ----------------------------

            track_id = int(
                box.id[0]
            )


            # ----------------------------
            # Bounding Box
            # ----------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ----------------------------
            # 클래스
            # ----------------------------

            class_id = int(
                box.cls[0]
            )

            class_name = result.names[
                class_id
            ]


            # ----------------------------
            # Confidence
            # ----------------------------

            confidence = float(
                box.conf[0]
            )


            # ----------------------------
            # 차량 중심점
            # ----------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ============================
            # 거리 추정
            # ============================

            box_height = y2 - y1

            box_ratio = box_height / height


            if box_ratio > 0.35:

                distance = "VERY CLOSE"

            elif box_ratio > 0.25:

                distance = "CLOSE"

            elif box_ratio > 0.15:

                distance = "MEDIUM"

            else:

                distance = "FAR"


            # ============================
            # 내 차선 근처인지 판단
            # ============================

            center_ratio = center_x / width

            in_front_area = (
                center_y > height * FRONT_Y_RATIO
            )

            near_center = (
                abs(center_ratio - LANE_CENTER)
                < 0.25
            )


            # ============================
            # 앞차 후보
            # ============================

            if in_front_area and near_center:

                if front_vehicle is None:

                    front_vehicle = {
                        "id": track_id,
                        "distance": distance,
                        "box_ratio": box_ratio,
                        "center_x": center_x,
                        "center_y": center_y
                    }

                else:

                    # 더 큰 차량 박스 =
                    # 더 가까운 차량
                    if box_ratio > front_vehicle[
                        "box_ratio"
                    ]:

                        front_vehicle = {
                            "id": track_id,
                            "distance": distance,
                            "box_ratio": box_ratio,
                            "center_x": center_x,
                            "center_y": center_y
                        }


            # ============================
            # Bounding Box
            # ============================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ============================
            # 차량 정보
            # ============================

            label = (
                f"ID {track_id} "
                f"{class_name} "
                f"{distance}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


            # ============================
            # 중심점
            # ============================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


    # ============================
    # 화면 중앙선
    # ============================

    center_line_x = int(
        width * LANE_CENTER
    )

    cv2.line(
        frame,
        (center_line_x, 0),
        (center_line_x, height),
        (255, 0, 0),
        2
    )


    # ============================
    # 앞차 표시
    # ============================

    if front_vehicle is not None:

        front_id = front_vehicle["id"]
        front_distance = front_vehicle["distance"]

        message = (
            f"FRONT VEHICLE: "
            f"ID {front_id} "
            f"| {front_distance}"
        )

        cv2.putText(
            frame,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # 앞차 중심점 강조

        cv2.circle(
            frame,
            (
                front_vehicle["center_x"],
                front_vehicle["center_y"]
            ),
            10,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "FRONT VEHICLE: NONE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ============================
    # 결과 출력
    # ============================

    cv2.imshow(
        "Driving Assist",
        frame
    )


    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# ============================
# 종료
# ============================

video.release()
cv2.destroyAllWindows()

print("Driving Assist 종료")
# python3 09_Driving_Assist/tracking.py