'''
전화번호부에 적힌 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인하려 합니다.
전화번호가 다음과 같을 경우, 구조대 전화번호는 영석이의 전화번호의 접두사입니다.
구조대 : 119
박준영 : 97 674 223
지영석 : 11 9552 4421
전화번호부에 적힌 전화번호를 담은 배열 phone_book 이 solution 함수의 매개변수로 주어질 때,
어떤 번호가 다른 번호의 접두어인 경우가 있으면 false를 그렇지 않으면 true를 return 하도록 solution 함수를 작성해주세요.
제한 사항
phone_book의 길이는 1 이상 1,000,000 이하입니다.
각 전화번호의 길이는 1 이상 20 이하입니다.
같은 전화번호가 중복해서 들어있지 않습니다.
'''
phone_book = list(map(str, input().split()))


def solution_1(phone_book):
    answer = True
    phone_book.sort(key=len)
    for i in range(len(phone_book) - 1):
        for j in range(i + 1, len(phone_book)):
            if phone_book[j][:len(phone_book[i])] == phone_book[i]:
                return False
    return answer


# 더 빠른 코드
def solution_2(phone_book):
    s = dict()
    for p in phone_book:
        for i in range(1, len(p)):
            s[p[:i]] = 1
    for p in phone_book:
        if p in s:
            return False
    return True


print(solution_1(phone_book))
print(solution_2(phone_book))
