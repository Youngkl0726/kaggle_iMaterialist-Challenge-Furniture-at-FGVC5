
# coding: utf-8

# In[2]:


import os
import sys


# In[25]:


basepath = 'valid'
valid = os.listdir(basepath)
dct = []
for label in valid:
    path = os.path.join(basepath,label)
    lst = os.listdir(path)
    A = []
    for x in lst:
        A.append(os.path.join(path,x))
    dct.append((int(label),A))
dct2 = sorted(dct,key=lambda x: x[0])
        


# In[27]:


with open("valid.txt","wb") as fw:
    for x in dct2:
        label = str(x[0]-1)
        A = x[1]
        for y in A:
            fw.write((y+' '+label+'\r\n').encode('utf-8'))


# In[28]:


with open('validnum.txt','wb') as fw:
    for x in dct2:
        label = str(x[0]-1)
        A = x[1]
        fw.write((label+','+str(len(A))+'\r\n').encode('utf-8'))

