import numpy as np
import csv

# average
npy = []
a = 0.0
b = 0.0
c = 1.0
print a,b,c
for i in xrange(3):
    print("i is: {}".format(i))
    npy_name = 'prob_dense{:0}.npy'.format(i)
    npy.append(np.load(npy_name))
# print npy[0][0], npy[1][0], npy[2][0]
npy_add = a*npy[0] + b*npy[1] + c*npy[2]
# print npy_add[0]
txt_file = open('result.txt', 'wb')
for i in xrange(12704):
    res = int(np.where(npy_add[i] == np.max(npy_add[i]))[0][0])
    txt_file.write(str(res)+'\n')
txt_file.close()


def get_name(filename):
    file = open(filename)
    res_line = []
    for i in xrange(12704):
        line = file.readline()
        line = line.strip()
        line = line.split(" ")
        line = line[0]
        line = line.split("/")
        line = line[1]
        line = line.split(".")
        # print line[0]
        res_line.append(line[0])
    return res_line

# def get_extra(filename):
#     file = open(filename)
#     res_line = []
#     for i in xrange(1):
#         line = file.readline()
#         line = line.strip()
#         line = line.split(" ")
#         for j in xrange(97):
#             # print line[j]
#             li = line[j]
#             li = li.split(".")
#             li = li[0]
#             res_line.append(li)
#     return res_line

def get_all(filename):
    file = open(filename)
    res_line = []
    for i in xrange(12800):
        line = file.readline()
        line = line.strip()
        # print line
        res_line.append(line)
    return res_line

fname = get_name("./data/test.txt")
all = get_all('whole_test.txt')
add = []
for i in xrange(12800):
    id = all[i]
    flag = 0
    for j in xrange(12704):
        if id == fname[j]:
            flag = 1
            break
    if flag == 0:
        add.append(id)
print add
# extra_name = get_extra("extra.txt")
# all_txt = open('whole_test.txt','wb')
# for i in xrange(12703):
#     all_txt.write(str(fname[i])+'\n')
# for i in xrange(97):
#     all_txt.write(str(extra_name[i])+'\n')
# add = []
# same = []
# num = 0
# for i in xrange(97):
#     name = extra_name[i]
#     flag = 0
#     for j in xrange(12703):
#         if name == fname[j]:
#             flag = 1
#             same.append(name)
#             break
#     if flag == 0:
#         add.append(name)
#         num+=1
# print num
# print same
# print fname

csvfile = open("result.csv", "w")
fileheader = ["id", "predicted"]
writer = csv.writer(csvfile)
writer.writerow(fileheader)
res_file = open('result.txt')
for i in xrange(12800):
    context = []
    if i < 12704:
        line = res_file.readline()
        line = line.strip()
        line = line.split(" ")
        context.append(fname[i])
        context.append(int(line[0])+1)
    else:
        context.append(add[i-12704])
        context.append("2")
    writer.writerow(context)
csvfile.close()


