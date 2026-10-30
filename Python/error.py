target_speed = 80
current_speed = 50

error = target_speed - current_speed

kp = 2

control = kp * error

print("목표 속도:", target_speed)
print("현재 속도:", current_speed)
print("오차:", error)
print("제어량:", control)

'''
if error>0:
    print("가속")
elif error<0:
    print("감속")
else:
    print("유지")
    '''