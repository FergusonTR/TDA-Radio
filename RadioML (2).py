
import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('T:/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')

xdset = f['X']
ydset = f['Y']
zdset = f['Z']

seq = 1500000

y_label = -1
classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']

x_seq = xdset[seq]
y_seq = ydset[seq]
z_seq = zdset[seq]

for y in range(24):
    if y_seq[y]==1:
        y_label = y
    
for x in range(1024):
    plt.scatter(x_seq[x,0],x_seq[x,1],s=10)

title_string = classes[y_label],'  ',z_seq[0]

plt.title(title_string, fontsize=14)

plt.ylabel("In-Phase",fontsize =10)
plt.xlabel("Quadrature",fontsize=10)

plt.show()

plt.scatter(x_seq)