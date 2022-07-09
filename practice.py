text = """A press release is the quickest and easiest way to get free publicity. If well written, a press release can result in multiple published articles about your firm and its prodects. And that can mean new prospects cantacting you asking you to sell to them.""".lower().split()

from collections import defaultdict

word_count = defaultdict(lambda: 0)
def order_by_value(t):
    return t[1]

for word in text:
    word_count[word] += 1

from collections import OrderedDict
for i, v in OrderedDict(sorted(word_count.items(), key=order_by_value, reverse=True)).items():
#for i, v in OrderedDict(sorted(word_count.items(), key=lambda t: t[1], reverse=True)).items():
    print(i, v)
