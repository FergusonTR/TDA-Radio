"""
Created on Fri Nov 12 07:51:20 2021
@author: Terry
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

#Set window size for samples which is consistent with RadioML 2018 dataset
window = 1024

#Open a noise file recorded from GnuRadio
samples = np.fromfile('/home/terry/2018.01/Noise_File.iq', np.complex64)
f_new = h5py.File("/home/terry/2018.01/mytestfile.hdf5", "w")

#Determine the size of the recorded array and split into a window of 1024 
split_value = samples.size//window

#determines the length of the array that is a multiple of the window
cut = split_value * window

#convert the recorded values to 32 bit floats
samples_I = np.real(samples[0:cut])
samples_Q = np.imag(samples[0:cut])

#Reshape the arrays to match the RadioML dataset
new = np.reshape(samples_I, (split_value, window))
new2 = np.reshape(samples_Q, (split_value, window))
arr = np.stack((new,new2),axis = -1)

#Create a dataset "X"
dset = f_new.create_dataset("X", data=arr)

#

#Close the files  
f_new.close()
samples.close()