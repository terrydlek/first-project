n = int(input())

count = 0
hour, minute, second = 0, 0, 0
while hour < (n + 1):
    second += 1
    if second == 60:
        second = 0
        minute += 1
        if minute == 60:
            minute = 0
            hour += 1
    clock = str(hour) + str(minute) + str(second)
    if "3" in clock:
        count += 1
print(count)