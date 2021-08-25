
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

#import csv
#import matplotlib.pyplot as plt
#from ripser import Rips
#from persim import PersImage
#from persim import PersistenceImager

f = h5py.File('c:/tmp/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
#csv_file = open('c:/tmp/csv_flle.csv', 'w')

xdset = f['X'] #'I-Q data points'
ydset = f['Y'] #'Modulation label'
zdset = f['Z'] #'SNR'

# seq = 246577

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

x_dataset = f['X'][...]
#print(x_dataset.size)
y_dataset = f['Y'][...]
#print(y_dataset.size)
z_dataset = f['Z'][...]
#print(z_dataset.size)

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.20, random_state=random_state)

print("process ended")
#csv_file.close()
f.close()