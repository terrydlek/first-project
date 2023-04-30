def func(a, b, c):
    if c != 0:
        func(b, a + b, c - 1)
        print(a, b, c)


func(1, 1, 3)
