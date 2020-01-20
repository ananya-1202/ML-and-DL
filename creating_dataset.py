import os
import matplotlib.pyplot as plt
import numpy as np
folder='C:/input_data'
folder2='C:/input_data_2'
b5='C:/library1'
b6='C:/ic'
b7='C:/shops'
b21='C:/library2'
b22='C:/nlh'
c5,c6,c7,c21,c22=os.listdir(b5),os.listdir(b6),os.listdir(b7),os.listdir(b21),os.listdir(b22)
imlist=os.listdir(folder)
p=0
n=len(imlist)
label=[]
i=0
for filename in os.listdir(b5):
        c5[i] =folder + '/' + filename
        i=i+1
        label.append('library1')
i=0
for filename in os.listdir(b6):
        c6[i] =folder + '/' + filename
        i=i+1
        label.append('ic')
i=0
for filename in os.listdir(b7):
        c7[i] =folder + '/' + filename
        i=i+1
        label.append('shops')
i=0
for filename in os.listdir(b21):
        c21[i] =folder + '/' + filename
        i=i+1
        label.append('library2')
i=0
for filename in os.listdir(b22):
        c22[i] =folder + '/' + filename
        i=i+1
        label.append('nlh')

training_data=[]
Data = c5+c6+c7+c21+c22+csp
for x in range(n):
    training_data.append([Data[x],label[x]])
