import numpy as np
import csv

# txt_file = open('val_result.txt', 'wb')
# npy_name = 'prob_0.npy'
# print npy_name
# npy = np.load(npy_name)
# # print npy[0],npy[1]
# # print np.where(npy[0] == np.max(npy[0]))[0][0]
# len = npy.shape[0]
# print len
# for j in xrange(len):
#     res = int(np.where(npy[j] == np.max(npy[j]))[0][0])
#     txt_file.write(str(res)+'\n')
# txt_file.close()

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
for i in xrange(6309):
    label = val_label[i]
    pre = pred[i]
    if label != pre:
        wrong_num[int(label)] += 1
        total+=1
print wrong_num
print total*1.0/6309.0