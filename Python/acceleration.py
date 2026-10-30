target_speed = 80
current_speed = 0

kp = 0.7
ki = 0.03
kd = 0.2
dt = 0.1

integral = 0
previous_error = 0

for i in range(50):
    
    error = target_speed - current_speed
    proportional = kp * error
    integral = integral + error * dt
    derivative = (error - previous_error) / dt
    derivative_control = kd * derivative

    control = proportional + ki * integral + derivative_control
    if control > 100:
        control = 100
    elif control < -100:
        control = -100

    current_speed = current_speed + control * dt

    previous_error = error

    print("시간:", i/10, "속도:", f"{current_speed:.2f}","오차:", f"{error:.2f}", "적분값:", f"{integral:.2f}")