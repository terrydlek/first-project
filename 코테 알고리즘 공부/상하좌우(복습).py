n = int(input())
plan = input().split()
x, y = 1, 1
for i in plan:
    if i == "D":
        if 1 <= x + 1 <= n:
            x += 1
    elif i == "U":
        if 1 <= x - 1 <= n:
            x -= 1
    elif i == "L":
        if 1 <= y - 1 <= n:
            y -= 1
    else:
        if 1 <= y + 1 <= n:
            y += 1

print(x, y)
