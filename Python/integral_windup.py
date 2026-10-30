from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from pathlib import Path

target_speed = 100
current_speed = 0

kp = 0.55
ki = 0.55
dt = 0.1
con = 25

integral = 0

# 엑셀에 저장할 데이터
data = []

for i in range(100):
    error = target_speed - current_speed
    proportional = kp * error
    control = proportional + ki * integral

    # 조건부 적분
    if control < con or error < 0:
        integral = integral + error * dt
    control = proportional + ki * integral

    # 제어량 제한
    if control > con:
        control = con

    current_speed = current_speed + control * dt

    # 데이터 기록
    data.append([
        round(i * dt, 1),
        target_speed,
        current_speed,
        error,
        integral,
        control
    ])

    print(
        "시간:", round(i * dt, 1),
        "속도:", round(current_speed, 2),
        "오차:", round(error, 2),
        "적분값:", round(integral, 2),
        "제어량:", round(control, 2)
    )





# 엑셀 파일 생성
wb = Workbook()
ws = wb.active
ws.title = "Simulation Data"

# 제목 입력
headers = ["Time", "Target Speed", "Current Speed",
           "Error", "Integral", "Control"]

ws.append(headers)

# 데이터 입력
for row in data:
    ws.append(row)

# 그래프 생성
chart = LineChart()
chart.title = "PID Speed Control"
chart.style = 13
chart.y_axis.title = "Speed"
chart.x_axis.title = "Time (s)"

# 목표 속도와 현재 속도 데이터 선택
values = Reference(ws, min_col=2, max_col=3,
                   min_row=1, max_row=ws.max_row)

# 시간 데이터 선택
times = Reference(ws, min_col=1,
                  min_row=2, max_row=ws.max_row)

chart.add_data(values, titles_from_data=True)
chart.set_categories(times)

chart.height = 10
chart.width = 20

# 그래프를 별도 시트에 배치
graph_ws = wb.create_sheet("PID Graph")
graph_ws.add_chart(chart, "A1")

# 파일 저장
save_path = Path(__file__).resolve().parent / "PID_result.xlsx"
wb.save(save_path)

print("엑셀 저장 완료:", save_path)
