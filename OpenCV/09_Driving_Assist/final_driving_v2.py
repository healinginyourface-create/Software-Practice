import cv2
import os
import math
import numpy as np
from ultralytics import YOLO
from collections import deque


# ============================================================
# 1. 경로 설정
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENCV_DIR = os.path.dirname(CURRENT_DIR)
VIDEO_PATH = os.path.join(OPENCV_DIR, "assets", "drivingvideo.mp4")
MODEL_PATH = os.path.join(OPENCV_DIR, "yolo11n.pt")


# ============================================================
# 2. 차량 클래스 (COCO)
# ============================================================

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


# ============================================================
# 3. YOLO 모델
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# 4. 카메라 파라미터 (핀홀 카메라 모델)
# ============================================================
#
# 주의: 이건 "진짜 캘리브레이션"이 아니라 근사치입니다.
# 진짜로 하려면 체스보드 패턴으로 cv2.calibrateCamera()를 써서
# focal_length와 왜곡계수(distortion coefficients)를 구해야 합니다.
# 여기서는 카메라 수평 화각(FOV)만 가정해서 focal length를 역산합니다.
#
# focal_length_px = (frame_width / 2) / tan(FOV_horizontal / 2)
#
# 블랙박스/대시캠 계열은 보통 수평 화각 60~90도 사이입니다.
# 실제 카메라 스펙을 알면 이 값을 정확히 넣어야 정확도가 올라갑니다.

CAMERA_FOV_HORIZONTAL_DEG = 78.0  # 대시캠 평균값으로 가정 (실측 필요)

# 차량 클래스별 평균 폭 (미터) — 이것도 근사치입니다.
VEHICLE_REAL_WIDTH_M = {
    2: 1.8,   # car
    3: 0.8,   # motorcycle
    5: 2.5,   # bus
    7: 2.5,   # truck
}


def compute_focal_length_px(frame_width, fov_deg):
    """
    수평 화각(FOV)으로부터 focal length(픽셀 단위)를 역산.
    진짜 캘리브레이션이 없을 때 쓰는 근사 공식.
    """
    fov_rad = math.radians(fov_deg)
    focal_length_px = (frame_width / 2.0) / math.tan(fov_rad / 2.0)
    return focal_length_px


def estimate_distance_m(box_width_px, class_id, focal_length_px):
    """
    핀홀 카메라 모델 기반 실제 거리(미터) 추정.

    distance = (real_world_width * focal_length_px) / pixel_width

    box_width_px가 너무 작으면(occlusion, 오탐 등) 비정상적으로 큰
    거리값이 나올 수 있으므로 하한선을 둔다.
    """
    if box_width_px <= 5:
        return None

    real_width = VEHICLE_REAL_WIDTH_M.get(class_id, 1.8)
    distance = (real_width * focal_length_px) / box_width_px
    return distance


def danger_level_from_distance(distance_m):
    if distance_m is None:
        return "UNKNOWN"
    if distance_m < 5:
        return "DANGER"
    elif distance_m < 15:
        return "CAUTION"
    else:
        return "SAFE"


# ============================================================
# 5. TTC 설정 (Lee's tau 기반)
# ============================================================
#
# TTC = size(t) / (d(size)/dt)
#
# size는 면적이 아니라 "선형 크기"(여기서는 박스 높이)를 씁니다.
# 면적은 거리의 제곱에 반비례하지만, 선형 크기는 거리에 정확히
# 반비례하기 때문에 물리적으로 더 정확합니다.
#
# 변화율(d(size)/dt)은 히스토리 전체에 대해 선형회귀(최소제곱법)로
# 구합니다 — 딱 두 프레임만 비교하는 것보다 노이즈에 훨씬 강합니다.

HISTORY_SIZE = 10
MIN_HISTORY_FOR_TTC = 5

TTC_DANGER = 2.0
TTC_CAUTION = 4.0

# track_id -> deque of (frame_index, box_height)
box_history = {}

# track_id -> 마지막으로 감지된 frame_index (메모리 정리용)
last_seen_frame = {}

# 이 프레임 수 이상 안 보이면 히스토리에서 삭제
TRACK_TIMEOUT_FRAMES = 30


def calculate_ttc(track_id, box_height, frame_index, fps):
    """
    선형회귀 기반 TTC 계산.

    1. (frame_index, box_height) 히스토리를 쌓는다.
    2. 최소제곱법으로 box_height의 시간에 따른 변화율(기울기)을 구한다.
    3. tau = 현재 크기 / 변화율(초당) 로 TTC를 계산한다.
    """

    if track_id not in box_history:
        box_history[track_id] = deque(maxlen=HISTORY_SIZE)

    history = box_history[track_id]
    history.append((frame_index, box_height))
    last_seen_frame[track_id] = frame_index

    if len(history) < MIN_HISTORY_FOR_TTC:
        return None

    frames = np.array([h[0] for h in history], dtype=np.float64)
    sizes = np.array([h[1] for h in history], dtype=np.float64)

    times = frames / fps  # 프레임 인덱스를 초 단위 시간으로 변환

    # 최소제곱법으로 1차 함수(size = a*t + b) 피팅
    # a: 초당 크기 변화율 (dw/dt)
    try:
        a, b = np.polyfit(times, sizes, 1)
    except np.linalg.LinAlgError:
        return None

    # 커지고 있지 않다면(멀어지거나 정지) TTC 의미 없음
    if a <= 0:
        return None

    current_size = sizes[-1]

    ttc = current_size / a

    # 비정상적으로 크거나 작은 값 필터링 (노이즈로 인한 튐 방지)
    if ttc <= 0 or ttc > 60:
        return None

    return ttc


def cleanup_old_tracks(current_frame_index):
    """사라진 지 오래된 track_id를 히스토리에서 정리 (메모리 누수 방지)."""
    stale_ids = [
        tid for tid, last_frame in last_seen_frame.items()
        if current_frame_index - last_frame > TRACK_TIMEOUT_FRAMES
    ]
    for tid in stale_ids:
        box_history.pop(tid, None)
        last_seen_frame.pop(tid, None)


# ============================================================
# 6. 차선 검출
# ============================================================

def detect_lane(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    height, width = edges.shape

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.60), int(height * 0.55)),
        (int(width * 0.40), int(height * 0.55))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180,
        threshold=30, minLineLength=40, maxLineGap=20
    )

    result = frame.copy()
    lane_lines = []

    if lines is not None:
        for line in lines:
            line = np.asarray(line).reshape(-1)
            if len(line) != 4:
                continue
            x1, y1, x2, y2 = line
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue
            lane_lines.append((x1, y1, x2, y2))
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    return result, lane_lines


# ============================================================
# 6-1. 차선 중앙 위치 계산 (+ EMA 스무딩)
# ============================================================

# 프레임 간 스무딩을 위한 전역 상태
_smoothed_lane_center = None
EMA_ALPHA = 0.2  # 낮을수록 더 부드럽지만 반응이 느려짐


def calculate_lane_center(lane_lines, frame_width, frame_height):

    global _smoothed_lane_center

    left_lines = []
    right_lines = []

    for line in lane_lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lines.append(line)
        elif slope > 0:
            right_lines.append(line)

    if not left_lines or not right_lines:
        # 검출 실패 시, 이전 스무딩 값을 그대로 유지 (완전히 놓치지 않도록)
        if _smoothed_lane_center is None:
            return None
        lane_center = _smoothed_lane_center
    else:
        def get_x_at_bottom(line):
            x1, y1, x2, y2 = line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return (frame_height - intercept) / slope

        left_x = np.mean([get_x_at_bottom(line) for line in left_lines])
        right_x = np.mean([get_x_at_bottom(line) for line in right_lines])
        raw_center = (left_x + right_x) / 2

        if _smoothed_lane_center is None:
            _smoothed_lane_center = raw_center
        else:
            _smoothed_lane_center = (
                EMA_ALPHA * raw_center
                + (1 - EMA_ALPHA) * _smoothed_lane_center
            )

        lane_center = _smoothed_lane_center

    vehicle_center = frame_width / 2
    offset = vehicle_center - lane_center
    tolerance = frame_width * 0.05

    if abs(offset) <= tolerance:
        status = "CENTER"
    elif offset > 0:
        status = "LEFT"
    else:
        status = "RIGHT"

    return (lane_center, vehicle_center, offset, status)


# ============================================================
# 7. 영상 초기화
# ============================================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("영상을 찾을 수 없습니다.")
    print(f"경로: {VIDEO_PATH}")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

frame_width_init = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
focal_length_px = compute_focal_length_px(frame_width_init, CAMERA_FOV_HORIZONTAL_DEG)

print("영상 불러오기 성공")
print(f"FPS: {fps:.1f}")
print(f"추정 focal length: {focal_length_px:.1f}px (FOV {CAMERA_FOV_HORIZONTAL_DEG}도 가정)")
print("========================================")
print("Driving Assist v2")
print("Lane(EMA) + YOLO + Tracking + Real Distance(Pinhole) + TTC(Lee's tau)")
print("========================================")

frame_index = 0


# ============================================================
# 8. 메인 루프
# ============================================================

while True:

    ret, frame = video.read()
    if not ret:
        break

    frame_index += 1
    height, width = frame.shape[:2]

    # ========================================================
    # A. Lane Detection
    # ========================================================

    result_frame, lane_lines = detect_lane(frame)
    lane_info = calculate_lane_center(lane_lines, width, height)

    if lane_info is not None:
        lane_center, vehicle_center, lane_offset, lane_status = lane_info

        cv2.circle(result_frame, (int(lane_center), height - 80), 8, (0, 255, 255), -1)
        cv2.line(result_frame, (int(vehicle_center), height - 120),
                  (int(vehicle_center), height), (255, 255, 255), 2)
        cv2.putText(result_frame, f"LANE: {lane_status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_frame, f"OFFSET: {lane_offset:.0f}px", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ========================================================
    # B. YOLO Tracking
    # ========================================================

    results = model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES)
    result = results[0]

    vehicles = []

    if result.boxes is not None:
        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            class_id = int(box.cls[0])

            if class_id not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_width = x2 - x1
            box_height = y2 - y1

            # ------------------------------------------------
            # 실제 거리 추정 (핀홀 카메라 모델)
            # ------------------------------------------------
            distance_m = estimate_distance_m(box_width, class_id, focal_length_px)

            # ------------------------------------------------
            # TTC (선형회귀 기반, box_height 사용)
            # ------------------------------------------------
            ttc = calculate_ttc(track_id, box_height, frame_index, fps)

            class_name = result.names[class_id]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            center_ratio = center_x / width
            center_difference = abs(center_ratio - 0.5)

            front_candidate = (
                center_y > height * 0.45
                and center_difference < 0.25
            )

            vehicles.append({
                "id": track_id,
                "class": class_name,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": center_x, "center_y": center_y,
                "distance_m": distance_m,
                "box_height": box_height,
                "front_candidate": front_candidate,
                "ttc": ttc,
            })

    # 사라진 트랙 정리
    cleanup_old_tracks(frame_index)

    # ========================================================
    # C. 가장 가까운 앞차 찾기 (거리 기준으로 변경)
    # ========================================================

    front_vehicle = None

    for vehicle in vehicles:
        if not vehicle["front_candidate"]:
            continue
        if vehicle["distance_m"] is None:
            continue

        if front_vehicle is None:
            front_vehicle = vehicle
        else:
            # 거리가 더 가까운 차량이 진짜 앞차
            if vehicle["distance_m"] < front_vehicle["distance_m"]:
                front_vehicle = vehicle

    # ========================================================
    # D. 차량 표시
    # ========================================================

    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["x1"], vehicle["y1"], vehicle["x2"], vehicle["y2"]
        track_id = vehicle["id"]
        class_name = vehicle["class"]
        confidence = vehicle["confidence"]
        distance_m = vehicle["distance_m"]
        ttc = vehicle["ttc"]

        is_front = (front_vehicle is not None and vehicle["id"] == front_vehicle["id"])
        box_color = (0, 0, 255) if is_front else (255, 0, 0)

        cv2.rectangle(result_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(result_frame, (vehicle["center_x"], vehicle["center_y"]), 5, (0, 255, 255), -1)

        label = f"ID {track_id} {class_name} {confidence:.2f}"
        cv2.putText(result_frame, label, (x1, max(y1 - 30, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if distance_m is not None:
            distance_text = f"{distance_m:.1f}m"
        else:
            distance_text = "-- m"
        cv2.putText(result_frame, distance_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        if ttc is None:
            ttc_text = "TTC: --"
        elif ttc < TTC_DANGER:
            ttc_text = f"TTC: {ttc:.1f}s DANGER"
        elif ttc < TTC_CAUTION:
            ttc_text = f"TTC: {ttc:.1f}s CAUTION"
        else:
            ttc_text = f"TTC: {ttc:.1f}s"

        cv2.putText(result_frame, ttc_text, (x1, min(y2 + 25, height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # ========================================================
    # E. 화면 중앙선
    # ========================================================

    center_x_screen = int(width * 0.5)
    cv2.line(result_frame, (center_x_screen, 0), (center_x_screen, height), (255, 0, 0), 1)

    # ========================================================
    # F. 위험도 판단 (거리 + TTC 결합)
    # ========================================================

    if front_vehicle is None:
        safety = "SAFE"
    else:
        safety = danger_level_from_distance(front_vehicle["distance_m"])
        front_ttc = front_vehicle["ttc"]

        if front_ttc is not None:
            if front_ttc < TTC_DANGER:
                safety = "DANGER"
            elif front_ttc < TTC_CAUTION and safety == "SAFE":
                safety = "CAUTION"

    if safety == "DANGER":
        status_text, status_color = "DANGER", (0, 0, 255)
    elif safety == "CAUTION":
        status_text, status_color = "CAUTION", (0, 165, 255)
    else:
        status_text, status_color = "SAFE", (0, 255, 0)

    # ========================================================
    # G. 상단 상태창
    # ========================================================

    cv2.rectangle(result_frame, (10, 10), (430, 110), (20, 20, 20), -1)
    cv2.putText(result_frame, f"STATUS: {status_text}", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    if front_vehicle is not None:
        front_id = front_vehicle["id"]
        front_distance = front_vehicle["distance_m"]
        front_ttc = front_vehicle["ttc"]

        distance_display = f"{front_distance:.1f}m" if front_distance is not None else "--"
        ttc_display = f"{front_ttc:.1f}s" if front_ttc is not None else "--"

        front_text = f"Front: ID {front_id} | {distance_display}"
        ttc_status_text = f"TTC: {ttc_display}"
    else:
        front_text = "Front Vehicle: NONE"
        ttc_status_text = "TTC: --"

    cv2.putText(result_frame, front_text, (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(result_frame, ttc_status_text, (25, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(result_frame, f"Vehicles: {len(vehicles)}", (width - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ========================================================
    # H. 출력 / 종료
    # ========================================================

    cv2.imshow("Driving Assist v2", result_frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("Driving Assist v2 종료")
print("========================================")