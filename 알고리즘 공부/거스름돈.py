n = int(input("받은 돈은?: ")) #거슬러 줘야 할 돈
count = 0
coin_types = [500, 100, 50, 10]
#큰 단위의 화폐부터 차례대로 확인
for coin in coin_types:
    count += n // coin #해당 화폐로 거슬러 줄 수 있는 동전의 개수 새기
    n %= coin
print(count)
