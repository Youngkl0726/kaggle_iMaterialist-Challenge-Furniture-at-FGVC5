file = open('result_resnext101_32x4d_ck4_label.txt')
test_label_txt = open('./data/test_label3.txt', 'wb')
for i in xrange(12704):
    line = file.readline()
    line = line.strip()
    line = line.split(",")
    name = line[0]
    id = int(line[1])-1
    # print name, id
    print 'test/test/{}.jpg'.format(name)+' '+str(id)+'\n'
    test_label_txt.write('test/test/{}.jpg'.format(name)+' '+str(id)+'\n')