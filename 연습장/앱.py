import googletrans


def translator_en(_text):
    # 영어로 변역
    trans = googletrans.Translator()
    result = trans.translate(_text, dest='en', src='auto', **kwargs)
    return result.text


def translator_ko(_text):
    # 한국어로 번역
    trans = googletrans.Translator()
    result = trans.translate(_text, dest='ko',src='auto')
    return result.text


print("ai 번역기를 선택학세요")
choice = input("1. 모든 언어 -> 영어, 2. 모든언어 -> 한국어: ")
if choice == "1":
    a = translator_en(input("입력 :"))
    print("번역:", a)
elif choice == "2":
    b = translator_ko(input("입력: "))
    print("번역:", b)
else:
    print("잘못 눌렀음")
