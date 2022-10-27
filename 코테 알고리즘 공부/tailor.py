from math import exp, sin, cos
import Module_2

if __name__ == "__main__":
    x = int(input("함수의 입력 x:"))
    N = int(input("테일러 급수의 차수 N: "))
    print("")

    print("exp(x)계산결과".center(20, '-'))
    print("math.exp(x): %.7f" % exp(x))
    print("테일러 근사: %.7f" % Module_2.exponential(x, N), end="\n\n")

    print("sin(x) 계산결과".center(20, '-'))
    print("math.sin(x): %.7f" % sin(x))
    print("테일러 근사: %.7f" % Module_2.sine(x, N), end="\n\n")

    print("cos(x) 계산결과".center(20, '-'))
    print("math cos(x): %7f" % cos(x))
    print("테일러 근사: %.7f" % Module_2.cosine(x, N))
#깃허브 연습
