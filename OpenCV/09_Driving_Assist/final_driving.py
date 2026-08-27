import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque


# ============================================================
# 1. 경로 설정
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OPENCV_DIR = os.path.dirname(
    CURRENT_DIR
)

VIDEO_PATH = os.path.join(
    OPENCV_DIR,
    "assets",
    "drivingvideo.mp4"
)

MODEL_PATH = os.path.join(
    OPENCV_DIR,
    "yolo11n.pt"
)


# ============================================================
# 2. 차량 클래스
# ============================================================

# COCO
# car        = 2
# motorcycle = 3
# bus        = 5
# truck      = 7

VEHICLE_CLASSES = [2, 3, 5, 7]


# ============================================================
# 3. YOLO 모델
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# 4. TTC 설정
# ============================================================

# 차량 ID별 Bounding Box 면적 기록
box_history = {}

# 최근 몇 프레임을 사용할지
HISTORY_SIZE = 10

# TTC 임계값
TTC_DANGER = 2.0
TTC_CAUTION = 4.0


# ============================================================
# 5. 영상
# ============================================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():

    print("영상을 찾을 수 없습니다.")
    print(f"경로: {VIDEO_PATH}")
    exit()


# 영상 FPS
fps = video.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30


print("영상 불러오기 성공")
print(f"FPS: {fps:.1f}")

print("========================================")
print("Nudge Autonomous Driving Perception")
print("Lane + YOLO + Tracking + Distance + TTC")
print("========================================")


# ============================================================
# 6. 차선 검출
# ============================================================

def detect_lane(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    height, width = edges.shape


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    mask = np.zeros_like(edges)

    polygon = np.array([
        [
            (0, height),
            (width, height),
            (
                int(width * 0.60),
                int(height * 0.55)
            ),
            (
                int(width * 0.40),
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


    # --------------------------------------------------------
    # Hough Transform
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=20
    )


    result = frame.copy()

    lane_lines = []


    if lines is not None:

        for line in lines:

            line = np.asarray(
                line
            ).reshape(-1)

            if len(line) != 4:
                continue

            x1, y1, x2, y2 = line


            if x2 == x1:
                continue


            slope = (
                (y2 - y1)
                /
                (x2 - x1)
            )


            # 너무 수평인 선 제거
            if abs(slope) < 0.5:
                continue


            lane_lines.append(
                (x1, y1, x2, y2)
            )


            cv2.line(
                result,
                (
                    int(x1),
                    int(y1)
                ),
                (
                    int(x2),
                    int(y2)
                ),
                (0, 255, 0),
                3
            )


    return result, lane_lines


# ============================================================
# 7. 상대 거리 추정
# ============================================================

def estimate_distance(
    box_height,
    frame_height
):

    ratio = (
        box_height
        /
        frame_height
    )


    if ratio > 0.35:

        return "VERY CLOSE", ratio

    elif ratio > 0.25:

        return "CLOSE", ratio

    elif ratio > 0.15:

        return "MEDIUM", ratio

    else:

        return "FAR", ratio


# ============================================================
# 8. 위험도 판단
# ============================================================

def danger_level(distance):

    if distance == "VERY CLOSE":

        return "DANGER"

    elif distance == "CLOSE":

        return "CAUTION"

    else:

        return "SAFE"


# ============================================================
# 9. TTC 계산
# ============================================================

def calculate_ttc(
    track_id,
    box_area,
    fps
):

    # --------------------------------------------------------
    # 차량별 기록 생성
    # --------------------------------------------------------

    if track_id not in box_history:

        box_history[track_id] = deque(
            maxlen=HISTORY_SIZE
        )


    history = box_history[
        track_id
    ]


    # 현재 Bounding Box 면적 저장
    history.append(
        box_area
    )


    # 데이터가 충분하지 않으면 계산하지 않음
    if len(history) < 5:

        return None


    # --------------------------------------------------------
    # 과거 / 현재 면적
    # --------------------------------------------------------

    old_area = history[0]

    new_area = history[-1]


    if old_area <= 0:

        return None


    # --------------------------------------------------------
    # 면적 증가율
    # --------------------------------------------------------

    growth_rate = (
        new_area - old_area
    ) / old_area


    # Bounding Box가 커지지 않는 경우
    if growth_rate <= 0:

        return None


    # --------------------------------------------------------
    # 실제 경과 시간
    # --------------------------------------------------------

    elapsed_frames = (
        len(history) - 1
    )

    elapsed_time = (
        elapsed_frames / fps
    )


    if elapsed_time <= 0:

        return None


    # --------------------------------------------------------
    # 초당 면적 증가율
    # --------------------------------------------------------

    area_growth_per_second = (
        growth_rate
        /
        elapsed_time
    )


    if area_growth_per_second <= 0:

        return None


    # --------------------------------------------------------
    # 근사 TTC
    # --------------------------------------------------------

    ttc = (
        1.0
        /
        area_growth_per_second
    )


    return ttc


# ============================================================
# 10. 메인 루프
# ============================================================

while True:


    # --------------------------------------------------------
    # 프레임 읽기
    # --------------------------------------------------------

    ret, frame = video.read()


    if not ret:

        break


    height, width = frame.shape[:2]


    # ========================================================
    # A. Lane Detection
    # ========================================================

    result_frame, lane_lines = detect_lane(
        frame
    )


    # ========================================================
    # B. YOLO Tracking
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        verbose=False,
        classes=VEHICLE_CLASSES
    )


    result = results[0]


    # ========================================================
    # C. 차량 정보
    # ========================================================

    vehicles = []


    if result.boxes is not None:


        for box in result.boxes:


            # ------------------------------------------------
            # Tracking ID 확인
            # ------------------------------------------------

            if box.id is None:

                continue


            track_id = int(
                box.id[0]
            )


            # ------------------------------------------------
            # 클래스
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            if class_id not in VEHICLE_CLASSES:

                continue


            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # Bounding Box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )


            # Bounding Box 면적
            box_area = (
                box_width
                *
                box_height
            )


            # ------------------------------------------------
            # TTC
            # ------------------------------------------------

            ttc = calculate_ttc(
                track_id,
                box_area,
                fps
            )


            # ------------------------------------------------
            # 클래스 이름
            # ------------------------------------------------

            class_name = result.names[
                class_id
            ]


            # ------------------------------------------------
            # 중심점
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # 거리 추정
            # ------------------------------------------------

            distance, box_ratio = (
                estimate_distance(
                    box_height,
                    height
                )
            )


            # ------------------------------------------------
            # 화면 중앙과의 거리
            # ------------------------------------------------

            center_ratio = (
                center_x
                /
                width
            )


            center_difference = abs(
                center_ratio - 0.5
            )


            # ------------------------------------------------
            # 앞차 후보
            # ------------------------------------------------

            front_candidate = (
                center_y > height * 0.45
                and
                center_difference < 0.25
            )


            # ------------------------------------------------
            # 차량 정보 저장
            # ------------------------------------------------

            vehicles.append({

                "id":
                    track_id,

                "class":
                    class_name,

                "confidence":
                    confidence,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "center_x":
                    center_x,

                "center_y":
                    center_y,

                "distance":
                    distance,

                "box_ratio":
                    box_ratio,

                "front_candidate":
                    front_candidate,

                "ttc":
                    ttc

            })


    # ========================================================
    # D. 가장 가까운 앞차 찾기
    # ========================================================

    front_vehicle = None


    for vehicle in vehicles:


        if not vehicle[
            "front_candidate"
        ]:

            continue


        if front_vehicle is None:

            front_vehicle = vehicle


        else:


            # Bounding Box가 큰 차량이
            # 카메라에 더 가까운 차량

            if vehicle[
                "box_ratio"
            ] > front_vehicle[
                "box_ratio"
            ]:

                front_vehicle = vehicle


    # ========================================================
    # E. 차량 Bounding Box 표시
    # ========================================================

    for vehicle in vehicles:


        x1 = vehicle["x1"]
        y1 = vehicle["y1"]
        x2 = vehicle["x2"]
        y2 = vehicle["y2"]

        track_id = vehicle["id"]

        class_name = vehicle["class"]

        confidence = vehicle["confidence"]

        distance = vehicle["distance"]

        ttc = vehicle["ttc"]


        # ----------------------------------------------------
        # 앞차 여부
        # ----------------------------------------------------

        is_front = (

            front_vehicle is not None

            and

            vehicle["id"]
            ==
            front_vehicle["id"]

        )


        # ----------------------------------------------------
        # 색상
        # ----------------------------------------------------

        if is_front:

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                255,
                0,
                0
            )


        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        cv2.rectangle(
            result_frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ----------------------------------------------------
        # 중심점
        # ----------------------------------------------------

        cv2.circle(
            result_frame,
            (
                vehicle["center_x"],
                vehicle["center_y"]
            ),
            5,
            (0, 255, 255),
            -1
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"ID {track_id} "
            f"{class_name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            result_frame,
            label,
            (
                x1,
                max(y1 - 30, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2
        )


        # ----------------------------------------------------
        # 거리
        # ----------------------------------------------------

        cv2.putText(
            result_frame,
            distance,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )


        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        if ttc is None:

            ttc_text = "TTC: --"

        elif ttc < TTC_DANGER:

            ttc_text = (
                f"TTC: {ttc:.1f}s DANGER"
            )

        elif ttc < TTC_CAUTION:

            ttc_text = (
                f"TTC: {ttc:.1f}s CAUTION"
            )

        else:

            ttc_text = (
                f"TTC: {ttc:.1f}s"
            )


        cv2.putText(
            result_frame,
            ttc_text,
            (
                x1,
                min(
                    y2 + 25,
                    height - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )


    # ========================================================
    # F. 화면 중앙선
    # ========================================================

    center_x = int(
        width * 0.5
    )


    cv2.line(
        result_frame,
        (
            center_x,
            0
        ),
        (
            center_x,
            height
        ),
        (255, 0, 0),
        1
    )


    # ========================================================
    # G. 위험도 판단
    # ========================================================

    if front_vehicle is None:

        safety = "SAFE"

    else:

        safety = danger_level(
            front_vehicle["distance"]
        )


        # TTC가 위험하면 위험도 상승
        front_ttc = front_vehicle["ttc"]


        if front_ttc is not None:

            if front_ttc < TTC_DANGER:

                safety = "DANGER"

            elif (
                front_ttc < TTC_CAUTION
                and
                safety == "SAFE"
            ):

                safety = "CAUTION"


    # ========================================================
    # H. 상태 표시
    # ========================================================

    if safety == "DANGER":

        status_text = "DANGER"

        status_color = (
            0,
            0,
            255
        )

    elif safety == "CAUTION":

        status_text = "CAUTION"

        status_color = (
            0,
            165,
            255
        )

    else:

        status_text = "SAFE"

        status_color = (
            0,
            255,
            0
        )


    # ========================================================
    # I. 상단 상태창
    # ========================================================

    cv2.rectangle(
        result_frame,
        (10, 10),
        (430, 110),
        (20, 20, 20),
        -1
    )


    cv2.putText(
        result_frame,
        f"STATUS: {status_text}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )


    # --------------------------------------------------------
    # 앞차 정보
    # --------------------------------------------------------

    if front_vehicle is not None:


        front_id = (
            front_vehicle["id"]
        )

        front_distance = (
            front_vehicle["distance"]
        )

        front_ttc = (
            front_vehicle["ttc"]
        )


        if front_ttc is None:

            ttc_display = "--"

        else:

            ttc_display = (
                f"{front_ttc:.1f}s"
            )


        front_text = (
            f"Front: ID {front_id} "
            f"| {front_distance}"
        )


        ttc_status_text = (
            f"TTC: {ttc_display}"
        )


    else:

        front_text = (
            "Front Vehicle: NONE"
        )

        ttc_status_text = (
            "TTC: --"
        )


    cv2.putText(
        result_frame,
        front_text,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        result_frame,
        ttc_status_text,
        (25, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # J. 차량 수
    # ========================================================

    cv2.putText(
        result_frame,
        f"Vehicles: {len(vehicles)}",
        (
            width - 180,
            40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # K. 결과 출력
    # ========================================================

    cv2.imshow(
        "Nudge Autonomous Driving",
        result_frame
    )


    # ========================================================
    # L. 종료
    # ========================================================

    if cv2.waitKey(30) & 0xFF == ord("q"):

        break


# ============================================================
# 종료
# ============================================================

video.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("Nudge Autonomous Driving 종료")
print("========================================")