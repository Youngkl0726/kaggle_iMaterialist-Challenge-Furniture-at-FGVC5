import numpy as np
import csv

txt_file = open('val_result.txt', 'wb')
npy_name = 'prob_val0.npy'
print npy_name
npy = np.load(npy_name)
# print npy[0],npy[1]
# print np.where(npy[0] == np.max(npy[0]))[0][0]
len = npy.shape[0]
print len
for j in xrange(len):
    res = int(np.where(npy[j] == np.max(npy[j]))[0][0])
    txt_file.write(str(res)+'\n')
txt_file.close()

wrong_num = []
for i in xrange(128):
    wrong_num.append(0)
# print wrong_num

def get_label(filename):
    file = open(filename)
    res_line = []
    for i in xrange(6309):
        line = file.readline()
        line = line.strip()
        line = line.split(" ")
        line = line[1]
        # print line
        res_line.append(line)
    return res_line

def get_pred(filename):
    file = open(filename)
    res_line = []
    for i in xrange(6309):
        line = file.readline()
        line = line.strip()
        # print line
        res_line.append(line)
    return res_line

val_label = get_label("./data/valid.txt")
pred = get_pred("val_result.txt")
total = 0
chair = [[] for i in range(4)]
id_list =[125, 28, 3, 62, 2, 25, 14, 22, 102]
# for i in xrange(3):
#     chair[i].append(0)
for i in xrange(6309):
    label = val_label[i]
    pre = pred[i]
    if label != pre:
        wrong_num[int(label)] += 1
        # print label, pre
        if int(label) == 62:
            chair[0].append(pre)
        if int(label) == 14:
            chair[1].append(pre)
        if int(label) == 3:
            chair[2].append(pre)
        if int(label) ==65:
            chair[3].append(pre)
        total+=1
        flag1 = False
        flag2 = False
        for j in xrange(9):
            if int(label) == id_list[j]:
                flag1 = True
            if int(pre) == id_list[j]:
                flag2 = True
        # if (flag1 == True or flag2 == True) and (flag1 != flag2):
        if flag1 == False and flag2 ==True:
            print label, pre
# for i in xrange(128):
#     print i, wrong_num[i]
sort_list =  sorted(enumerate(wrong_num), key = lambda x:x[1], reverse=True)
# print sort_list
# print set(chair[0]), chair[0]
# print set(chair[1]), chair[1]
# print set(chair[2]), chair[2]
# print chair[3]

print total*1.0/6309.0