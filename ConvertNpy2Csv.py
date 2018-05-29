import numpy as np
import csv

txt_file = open('result.txt', 'wb')
npy_name = 'nasnet_ck2.npy' #change the npy_name
print npy_name
npy = np.load(npy_name)
len = npy.shape[0]
print len
for j in xrange(len):
    res = int(np.where(npy[j] == np.max(npy[j]))[0][0])
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

fname = get_name("test.txt")
add = []

csvfile = open("result_nasnet_ck2.csv", "w") #change the name of files you will save
fileheader = ["id", "predicted"]
writer = csv.writer(csvfile)
writer.writerow(fileheader)
res_file = open('result.txt')
for i in xrange(12704):
    context = []
    line = res_file.readline()
    line = line.strip()
    line = line.split(" ")
    context.append(fname[i])
    context.append(int(line[0])+1)
    writer.writerow(context)
csvfile.close()
