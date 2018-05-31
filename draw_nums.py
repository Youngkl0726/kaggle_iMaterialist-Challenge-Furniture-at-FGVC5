import matplotlib.pyplot as plt
import numpy as np
import csv

filename = 'result_resnext101_32x4d_ck4.csv'
nums = np.zeros(128)
with open(filename) as f:
    reader = csv.reader(f)
    head_row = next(reader)
    for row in reader:
        nums[int(row[1])-1] += 1
print(nums)

plt.figure(figsize=(20, 5))
plt.plot(range(0,128),nums) 
plt.xlabel("Label")
plt.ylabel("Sum")
plt.title("Label-Sum")
plt.show()

plt.figure(figsize=(20, 5))
plt.hist(nums,128) 
plt.xlabel("Sum of a label")
plt.ylabel("The numbers of the same")
plt.title("data distribution")
plt.show()

