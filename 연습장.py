graph = [[1, 1, 1], [0, 0, 1], [1, 1, 1]]
vi = [[False, False, False], [False, False, False], [False, False, False]]
for i in range(3):
    for j in range(3):
        if graph[i][j] == 1:
            vi[i][j] = True
print(vi)
'''
li = [[0]*5]
print(li)
li.append([0,2])
print(li)'''