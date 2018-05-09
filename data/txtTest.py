
# coding: utf-8

# In[1]:


import os
import sys


# In[9]:


basepath = 'test'
test = os.listdir(basepath)
A = []
for x in test:
    A.append(os.path.join(basepath,x))
dct = sorted(A,key=lambda x: x[0])
print(dct)


# In[14]:


with open("test.txt","wb") as fw:
    for x in dct:
        x = x.replace("\\","/")
        fw.write((x+' '+str(0)+'\r\n').encode('utf-8'))

