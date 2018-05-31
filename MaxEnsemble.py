import numpy as np
import csv

model_list = ['resnext101_32x4d_ck2.npy', 'dpn131_ck2.npy']

length = len(model_list)
# average
npy = []
for i in xrange(0, length):
    model_name = model_list[i]
    print("model name is: {}".format(model_name))
    npy_file = '/Users/youngkl/Desktop/fur_pse/'+model_name
    npy.append(np.load(npy_file))

txt_file = open('result.txt', 'wb')
for i in xrange(12704):
    max_prob = -1.0
    res_id = -2
    for j in xrange(length):
        cur_npy = npy[j]
        # print cur_npy[i]
        id = int(np.where(cur_npy[i] == np.max(cur_npy[i]))[0][0])
        # print("id is: {}".format(id))
        prob = cur_npy[i][id]
        if prob > max_prob:
            max_prob = prob
            res_id = id
        # print("prob is: {}".format(prob))
    # print("res_id is: {}".format(res_id))
    # print("max_prob is: {}".format(max_prob))
    txt_file.write(str(res_id)+'\n')
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

csvfile = open("Max_ensemble1.csv", "w")
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

# max_ensemble1
# model_list = ['resnext101_32x4d_ck2.npy', 'dpn131_ck2.npy']  0.13281