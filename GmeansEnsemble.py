import numpy as np
import csv

model_list = ['resnext101_32x4d_ck2.npy', 'dpn131_ck2.npy', 'dpn107_ck4.npy', \
              'resnext101_32x4d_ck4.npy', 'dpn92_ck6.npy', 'dpn98_ck6.npy',\
              'dpn92_ck3.npy', 'dpn98_ck3.npy', \
              'inceptionresnetv2_ck3.npy', 'inceptionv4_ck3.npy',
              'resnet152_ck8.npy', 'senet154_ck3.npy', 'resnext101_64x4d_ck2.npy',\
              'se_resnet152_ck2.npy', 'dpn92_ck5.npy', 'dpn131_ck5.npy',\
              'resnet152_ck10.npy', 'se_resnet152_ck4.npy', 'senet154_ck5.npy',\
              'resnet152_ck11.npy', 'resnext101_32x4d_ck5.npy', 'dpn131_ck6.npy'] # best ensemble

length = len(model_list)

npy = []
for i in xrange(0, length):
    model_name = model_list[i]
    print("model name is: {}".format(model_name))
    npy_file = '/Users/youngkl/Desktop/fur_pse/'+model_name
    npy.append(np.load(npy_file))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

for i in xrange(length):
    pre = npy[i]
    for j in xrange(12704):
        x = pre[j]
        y = softmax(x)
        pre[j] = y
    npy[i] = pre
# print npy[0]

npy_res = npy[0]
for i in xrange(1, length):
    npy_res = npy_res*npy[i]

txt_file = open('result.txt', 'wb')
for i in xrange(12704):
    npy_res[i] = pow(npy_res[i], 1.0/(length*1.0))
    res = int(np.where(npy_res[i] == np.max(npy_res[i]))[0][0])
    txt_file.write(str(res)+'\n')
txt_file.close()


# npy_res = npy[0]
# for i in xrange(10):
#     x = npy_res[i]
#     y = softmax(x)
#     print y[20]
#     res = int(np.where(y == np.max(y))[0][0])
#     print res



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

csvfile = open("ensemble.csv", "w")
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
