import numpy as np
import csv

model_list = ['res152_ck5.npy', 'inceptionResnetv2_ck1.npy', 'dpn98_ck1.npy', \
              'senet154_ck2.npy', 'dpn131_ck1.npy', 'inceptionResnetv2_ck2.npy', \
              'res152_ck7.npy', 'nasnet_ck1.npy', 'dpn98_ck2.npy']

# average
npy = []
for i in xrange(0,9):
    model_name = model_list[i]
    print("model name is: {}".format(model_name))
    npy_file = '/Users/youngkl/Desktop/fur_res/'+model_name
    npy.append(np.load(npy_file))
# print npy[0][0], npy[1][0], npy[2][0]
npy_res = npy[0]
for i in xrange(1,9):
    npy_res = npy_res + npy[i]
# print npy_add[0]
txt_file = open('result.txt', 'wb')
for i in xrange(12704):
    res = int(np.where(npy_res[i] == np.max(npy_res[i]))[0][0])
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


