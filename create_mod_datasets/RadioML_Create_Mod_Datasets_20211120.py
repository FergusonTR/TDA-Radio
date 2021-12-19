#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:43:28 2021

@author: terry
 Leverages code written by Alexi Vaner for his Deep Learning Base Radio Signal Classification
"""
import numpy as np
import h5py

#f = h5py.File('c:/tmp/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
f = h5py.File('/home/terry/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
dir_path = r'/home/terry/2018.01/ExtractDataset'
#modu_snr_size = 1200
print("Opened File.....")

classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK',
           '32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM',
           '256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
           'GMSK','OQPSK','Noise']

#Creates Files of each modulation Mode
for modu in range(24):
	X_list = []
	Y_list = []
	Z_list = []  
	print('modu_'+classes[modu])
	start_modu = modu*106496
	end_modu = start_modu + 106496       
	X_list.append(f['X'][start_modu:end_modu])
	Y_list.append(f['Y'][start_modu:end_modu])
	Z_list.append(f['Z'][start_modu:end_modu])
	
	filename = dir_path + '/mod_' + classes[modu]+ '.h5'
	fw = h5py.File(filename,'w')
	fw['X'] = np.vstack(X_list)
	fw['Y'] = np.vstack(Y_list)
	fw['Z'] = np.vstack(Z_list)
	print('X shape:',fw['X'].shape)
	print('Y shape:',fw['Y'].shape)
	print('Z shape:',fw['Z'].shape)
	fw.close()
f.close()


