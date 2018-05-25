def get_name(filename):
    file = open(filename)
    res_line = []
    for i in xrange(12704):
        line = file.readline()
        line = line.strip()
        line = line.split(" ")
        # print line[0]
        res_line.append(line[0])
    return res_line

fname = get_name('./data/test.txt')
# print fname
def get_test_pre(filename):
    file = open(filename)
    res_line = []
    for i in xrange(12704):
        line = file.readline()
        line = line.strip()
        # line = line.split(" ")
        # print line[0]
        res_line.append(int(line))
    return res_line
test_pre = get_test_pre("result.txt")
# print test_pre
chair_test_file = open('chair_test.txt', 'wb')
id_list =[125, 28, 3, 62, 2, 25, 14, 22, 102]
for i in xrange(12704):
    for j in xrange(9):
        if test_pre[i] == id_list[j]:
            chair_test_file.write(fname[i]+" "+str(i)+'\n')
            break
