import numpy as np
import csv

txt_file = open('result.txt', 'wb')
npy_name = 'prob_dense0.npy'
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

#
# txt_file = open('result.txt', 'wb')
# npy_name = 'prob_test_1.npy'
# print npy_name
# npy = np.load(npy_name)
# len = npy.shape[0]
# print len
# for j in xrange(len):
#     res = int(np.where(npy[j] == np.max(npy[j]))[0][0])
#     txt_file.write(str(res)+'\n')
# npy_name2 = 'prob_test_2.npy'
# print npy_name2
# npy2 = np.load(npy_name2)
# len2 = npy2.shape[0]
# print len2
# for j in xrange(len2):
#     res = int(np.where(npy2[j] == np.max(npy2[j]))[0][0])
#     txt_file.write(str(res)+'\n')
# txt_file.close()

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
