import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import umap

#Open RadioML file modulation 
#f = h5py.File('/home/terry/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
f = h5py.File('/home/terry/2018.01/ExtractDataset/mod_4ASK.h5','r')

#Corrected Classes not from dataset with noise added
classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK',
           '32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM',
           '256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
           'GMSK','OQPSK','Noise']
           
#load data into np arrays
f1 = f['X'][:] #Array of I-Q data
f2 = f['Y'][:] #Array of Modulation modes
f3 = f['Z'][:] #Array of SNR data

#This Section is for testing lookup of plots
xdset = f['X']  #'I-Q data points'
ydset = f['Y']  #'Modulation Mode'
zdset = f['Z']  #'Noise Floor'
    
noise = 1           #1-26 (-20 to 30 dB 2dB increments)
modulation = 0       #0-23 classes in order listed below
index = 4096         #1-4096 in a single block
offset = 4096        #static block offset

seq = (106496 * modulation) + (noise * 4096) + (index - offset) - 1

print(seq)

y_label = -1
x_seq = xdset[seq]
y_seq = ydset[seq]
z_seq = zdset[seq]

for y in range(24):
    if y_seq[y]==1:
        y_label = y

filename = classes[y_label] + '_' + str(z_seq[0]) +'.png'

Magnitude = np.sqrt((x_seq[:,0] * x_seq[:,0]) + (x_seq[:,1] * x_seq[:,1]))

plt.plot(x_seq[:,0],color ='blue',linewidth=1, linestyle='solid')
plt.plot(x_seq[:,1],color ='green',linewidth=1, linestyle='solid')
title_string = classes[y_label],'  ',z_seq[0]
plt.title(title_string, fontsize=14)
plt.show()
#plt.savefig(my_path +'IQ_'+ filename)
#plt.clf()

plt.plot(Magnitude,color ='black',linewidth=1, linestyle='solid')

title_string = classes[y_label],'  ',z_seq[0]
plt.title(title_string, fontsize=14)
plt.ylabel("Magitude",fontsize =10)
plt.xlabel("Sample",fontsize=10)
plt.show()
#plt.savefig(my_path + 'Mag_'+ filename)
#plt.clf()

plt.scatter(x_seq[:,0],x_seq[:,1],s=5)

title_string = classes[y_label],'  ',z_seq[0]
plt.title(title_string, fontsize=14)
plt.xlabel("In-phase",fontsize =10)
plt.ylabel("Quadrature",fontsize=10)
plt.show()
#plt.savefig(my_path + 'const_'+ filename)
#plt.clf()
	
f.close()

