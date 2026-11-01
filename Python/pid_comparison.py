target_speed = 100
dt = 0.1
steps = 300
target_speed = 100
dt = 0.1
steps = 300

# PID 제어 계수
kp = 0.6
ki = 0.2
kd = 0.2

# 각 제어기의 초기 속도
p_speed = 0
pi_speed = 0
pid_speed = 0

# 각 제어기의 적분값
pi_integral = 0
pid_integral = 0

# PID의 이전 오차
previous_error = 0

# 제어기별 데이터 저장
p_data = []
pi_data = []
pid_data = []
pid_integral_data = []

    
for i in range(steps):
    # 1. P 제어
    p_error = target_speed - p_speed
    p_control = kp * p_error

    p_speed += p_control * dt
    p_data.append(p_speed)

    # 2. PI 제어
    pi_error = target_speed - pi_speed
    pi_integral += pi_error * dt

    pi_control = kp * pi_error + ki * pi_integral

    pi_speed += pi_control * dt
    pi_data.append(pi_speed)

    # 3. PID
    pid_error = target_speed - pid_speed

    pid_integral += pid_error * dt
    pid_integral_data.append(pid_integral)

    derivative = (pid_error - previous_error) / dt

    pid_control = (
        kp * pid_error
        + ki * pid_integral
        + kd * derivative
    )

    pid_speed += pid_control * dt
    pid_data.append(pid_speed)

    previous_error = pid_error

def calculate_performance(data, target, dt):

    # 상승 시간
    t_10 = next(
        (i * dt for i, speed in enumerate(data)
         if speed >= target * 0.1),
        None
    )

    t_90 = next(
        (i * dt for i, speed in enumerate(data)
         if speed >= target * 0.9),
        None
    )

    if t_10 is None or t_90 is None:
        rise_time = None
    else:
        rise_time = t_90 - t_10

    # 오버슈트
    max_speed = max(data)
    overshoot = max(0, (max_speed - target) / target * 100)

    # 정착 시간
    last_outside = max(
        (i for i, speed in enumerate(data)
         if abs(speed - target) > target * 0.02),
        default=-1
    )

    settling_time = (last_outside + 1) * dt

    if last_outside == len(data) - 1:
        settling_time = None

    return rise_time, overshoot, settling_time

# 제어기별 성능 출력
for name, data in [
    ("P", p_data),
    ("PI", pi_data),
    ("PID", pid_data)
]:
    rise, over, settling = calculate_performance(
        data, target_speed, dt
    )

    print(f"\n{name} 제어")

    if rise is not None:
        print(f"상승 시간: {rise:.2f}초")
    else:
        print("상승 시간: 측정 불가")

    print(f"오버슈트: {over:.2f}%")

    if settling is not None:
        print(f"정착 시간: {settling:.2f}초")
    else:
        print("정착 시간: 미정")


import matplotlib.pyplot as plt

time = [i * dt for i in range(steps)]

plt.plot(time, p_data, label="P")
plt.plot(time, pi_data, label="PI")
plt.plot(time, pid_data, label="PID")

plt.axhline(y=target_speed, color="black",
            linestyle="--", label="Target Speed")

plt.xlabel("Time (s)")
plt.ylabel("Speed")
plt.title("P vs PI vs PID Control")
plt.legend()
plt.grid()

plt.show()

print("시간 데이터:", len(time))
print("적분 데이터:", len(pid_integral_data))

plt.figure(figsize=(10, 5))

plt.plot(time, pid_integral_data, label="PID Integral")
plt.axhline(0, color="black", linestyle="--")

plt.xlabel("Time (s)")
plt.ylabel("Integral Value")
plt.title("PID Integral Term")
plt.legend()
plt.grid(True)

plt.show()