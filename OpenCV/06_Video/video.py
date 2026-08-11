import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(BASE_DIR, "assets", "car.jpg")

image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 결과 저장 폴더
output_dir = os.path.join(BASE_DIR, "frames")
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while frame_count < 10:

    # 프레임
    frame = image.copy()

    frame_count += 1

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny Edge
    edges = cv2.Canny(gray, 50, 150)

    # 결과 저장
    output_path = os.path.join(
        output_dir,
        f"frame_{frame_count:02d}.jpg"
    )

    cv2.imwrite(output_path, edges)

    print(f"Frame {frame_count} 처리 완료")

print("영상 처리 완료")
print(f"결과 저장 위치: {output_dir}")