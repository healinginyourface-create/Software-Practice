import cv2
import os


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


# 영상 읽기
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("영상을 찾을 수 없습니다.")
    print(f"확인한 경로: {video_path}")
    exit()

print("영상 불러오기 성공")


# ============================
# 영상 처리
# ============================

while True:

    # 프레임 읽기
    ret, frame = video.read()

    if not ret:
        break


    # 1. Grayscale

    gray = cv2.cvtColor(
        frame,
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


    # 4. 큰 Contour만 Bounding Box 표시

    object_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # 너무 작은 영역 제거
        if area < 100:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Bounding Box

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        object_count += 1


    # 객체 수 표시

    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # 결과 화면

    cv2.imshow(
        "Object Detection",
        frame
    )


    # 속도 조절 + Q 종료

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# 종료

video.release()
cv2.destroyAllWindows()

print("Object Detection 종료")