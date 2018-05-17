
# coding: utf-8

# In[2]:


import re
import os
import matplotlib.pyplot as plt


# In[3]:


log_path = "./log/log6.txt"
soft = True
ep = 77


# In[4]:


val_prec1=[]
val_loss=[]
train_prec1=[]
train_loss=[]
with open(log_path, 'r') as fp:
    Iter = 0
    Iter2 = 0
    for line in fp:
        m1 = re.search(r'Prec@1 ([0-9.,]+)	Loss ([0-9.,]+)',line)
        m2 = re.search(r'Data [0-9.,]+ \([0-9.,]+\)	Loss ([0-9.,]+) \(([0-9.,]+)\)	Prec@1 ([0-9.,]+) \(([0-9.,]+)\)',line)
        if m1:
            s1 = float(m1.group(1).replace(',', ''))
            s2 = float(m1.group(2).replace(',', ''))
            val_prec1.append(s1)
            val_loss.append(s2)
            Iter += 1
        if m2:
            if soft and (Iter2 % ep == 0) and (Iter2 != 0):
                train_loss.append(train_loss[Iter2-1])
                train_prec1.append(train_prec1[Iter2-1])
                Iter2 += 1
            else:
                s1 = float(m2.group(2).replace(',', ''))
                s2 = float(m2.group(4).replace(',', ''))
                train_loss.append(s1)
                train_prec1.append(s2)
                Iter2 += 1


# In[5]:


plt.figure(figsize=(15, 30))
plt.subplot(411)
plt.plot(range(len(val_loss)), val_loss, label='val_loss')
plt.title('Val_loss over '+ str(len(val_loss)) +' Epochs', size=15)
plt.xlabel("Epochs")
plt.ylabel("val_loss")
plt.legend()
plt.grid(True)

plt.subplot(412)
plt.plot(range(len(train_loss)), train_loss, label='train_loss')
plt.title('Train_loss over '+ str(len(train_loss)) +' Epochs', size=15)
plt.xlabel("Epochs")
plt.ylabel("train_loss")
plt.legend()
plt.grid(True)

plt.subplot(413)
plt.plot(range(len(val_prec1)), val_prec1, label='val_prec1')
plt.title('Val_prec1 over '+ str(len(val_prec1)) +' Epochs', size=15)
plt.xlabel("Epochs")
plt.ylabel("val_prec1")
plt.legend()
plt.grid(True)

plt.subplot(414)
plt.plot(range(len(train_prec1)), train_prec1, label='train_prec1')
plt.title('Train_prec1 over '+ str(len(train_prec1)) +' Epochs', size=15)
plt.xlabel("Epochs")
plt.ylabel("train_prec1")
plt.legend()
plt.grid(True)
plt.savefig("loss.png")
plt.show()

