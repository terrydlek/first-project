#주사위 두개를 36,000번 던져서 나오는 모든 경우의 수를 계산하는 프로그램
from random import randint
store = [0] * 11

for _ in range(36000):
    first_dice = randint(1, 6)
    second_dice = randint(1, 6)
    result = first_dice + second_dice
    store[result - 2] += 1

[print("{} : {} ({})".format(i+2, store[i], store[i]/36000)) for i in range(11)]
