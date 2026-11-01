import matplotlib.pyplot as plt

# -----------------------------
# 1. 차량 기본 설정
# -----------------------------

mass = 1500

air_density = 1.225
drag_coefficient = 0.3
frontal_area = 2.2
rolling_resistance = 0.01
gravity = 9.81

# -----------------------------
# 2. 제어 목표 및 초기 상태
# -----------------------------

target_speed = 100 / 3.6  # 100km/h -> m/s

speed = 0
dt = 0.1

max_driving_force = 5000

# -----------------------------
# 3. PID 제어기 설정
# -----------------------------

Kp = 500
Ki = 30
Kd = 600

integral = 0
previous_speed = speed

# -----------------------------
# 4. 데이터 저장
# -----------------------------

time_data = []
speed_data = []
force_data = []

# -----------------------------
# 5. 시뮬레이션
# -----------------------------

for i in range(300):

    # 현재 속도 오차
    error = target_speed - speed

    # P 제어
    P = Kp * error

    # I 제어
    I = Ki * integral

    # D 제어
    D = -Kd * (speed - previous_speed) / dt

    # PID 출력: 필요한 구동력
    control_force = P + I + D

    # 구동력 제한
    driving_force = max(
        0,
        min(control_force, max_driving_force)
    )

    # Anti-Windup: 구동력이 제한되지 않을 때만 적분
    if 0 < control_force < max_driving_force:
        integral += error * dt

    # 적분값 갱신 후 PID 출력 재계산
    I = Ki * integral
    control_force = P + I + D

    driving_force = max(
        0,
        min(control_force, max_driving_force)
    )

    # 공기저항
    drag_force = (
        0.5 * air_density * drag_coefficient
        * frontal_area * speed**2
    )

    # 구름저항
    rolling_force = rolling_resistance * mass * gravity

    # 전체 힘
    net_force = driving_force - drag_force - rolling_force

    # 가속도
    acceleration = net_force / mass

    # 속도 업데이트
    previous_speed = speed
    speed += acceleration * dt

    # 데이터 저장
    time_data.append((i + 1) * dt)
    speed_data.append(speed * 3.6)
    force_data.append(driving_force)


# -----------------------------
# 6. 그래프 출력
# -----------------------------

plt.plot(time_data, speed_data, label="Actual Speed")
plt.axhline(
    target_speed * 3.6,
    color="red",
    linestyle="--",
    label="Target Speed"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (km/h)")
plt.title("PID Vehicle Speed Control")
plt.legend()
plt.grid()

plt.show()