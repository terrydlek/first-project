def factorial(N):
    if N <= 1:
        return N
    else:
        return N * factorial(N-1)

def exponential(x,N):
    if N == 0:
        return 1
    else:
        return (x**N)/factorial(N) + exponential(x, N-1)

def sine(x, N):
    if N < 0:
        return 0
    else:
        return ((x**(2*N+1)) * (-1)**N)/factorial((2*N+1)) + sine(x, N-1)

def cosine(x, N):
    if N < 1:
        return 1
    else:
        return ((x**(2*N)) * (-1)**N)/factorial((2*N)) + cosine(x, N-1)
