"""
ori_id, chair_id
125, 0
28,  1
3,   2
62,  3
2,   4
25,  5
14,  6
22,  7
102, 8
"""
id_list =[125, 28, 3, 62, 2, 25, 14, 22, 102]
train_txt = open("./data/valid.txt")
chair_train_txt = open("./data/chair_valid.txt", 'wb')
num = []
for i in xrange(9):
    num.append(0)
for i in xrange(6309):
    line = train_txt.readline()
    line = line.strip()
    li = line.split(" ")
    id = int(li[1])
    for j in xrange(9):
        if id == id_list[j]:
            num[j]+=1
            chair_train_txt.write(li[0]+" "+str(j)+"\n")
chair_train_txt.close()
print num
# [1922, 1416, 1479, 1733, 2355, 1549, 1053, 1178, 2186]
# [48, 49, 49, 50, 49, 47, 47, 49, 49]