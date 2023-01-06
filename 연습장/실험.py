from collections import Counter
a = "park je uk"
b = Counter(a)
print(b)
b.pop("r")
print(b)