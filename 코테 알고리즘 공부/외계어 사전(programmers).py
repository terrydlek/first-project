'''
PROGRAMMERS-962 행성에 불시착한 우주비행사 머쓱이는 외계행성의 언어를 공부하려고 합니다.
알파벳이 담긴 배열 spell과 외계어 사전 dic이 매개변수로 주어집니다.
spell에 담긴 알파벳을 한번씩만 모두 사용한 단어가 dic에 존재한다면 1, 존재하지 않는다면 2를 return하도록 solution 함수를 완성해주세요.
'''
spell = list(map(str, input().split()))
dic = list(map(str, input().split()))


def solution_1(spell, dic):
    result = 0
    for i in dic:
        count = 0
        for j in i:
            if j in spell and i.count(j) == 1:
                count += 1
        if count == len(spell):
            result += 1
    if result >= 1:
        answer = 1
    else:
        answer = 2
    return answer


def solution_2(spell, dic):
    spell = set(spell)
    for s in dic:
        if not spell-set(s):
            return 1
    return 2


def solution_3(spell, dic):
    for d in dic:
        # sorted함수를 사용하면 무조건 리스트로 반환!!!
        if sorted(d) == sorted(spell):
            return 1
    return 2


print(solution_1(spell, dic))
print(solution_2(spell, dic))
print(solution_3(spell, dic))
