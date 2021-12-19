"""
Created on Fri Nov 12 07:51:20 2021
@author: Terry
"""
import h5py
import numpy as np

# Set window size for samples which is consistent with RadioML 2018 dataset
window = 1024

# Open a noise file recorded from GnuRadio
# Change these file names for whatever noise file you want to process
samples = np.fromfile('/home/terry/2018.01/Noise_File.iq', np.complex64)
f_new = h5py.File("/home/terry/2018.01/mytestfile.hdf5", "w")

#Determine the size of the recorded array and split into a window of 1024 
split_value = samples.size//window

# determines the length of the array that is a multiple of the window
cut = split_value * window

# convert the recorded values to 32 bit floats
samples_I = np.real(samples[0:cut])
samples_Q = np.imag(samples[0:cut])

# Reshape the arrays to match the RadioML dataset
samples_I = np.reshape(samples_I, (split_value, window))
samples_Q = np.reshape(samples_Q, (split_value, window))
arr = np.stack((samples_I,samples_Q),axis = -1)

# Create a dataset "X" with the random noise
dsetX = f_new.create_dataset("X", data=arr)

del dsetX
del samples_I
del samples_Q

# Create dataset "Y" and "Z" with modulation modes all set to zero and a 25th 
# noise marker and a SNR array with -50
noise_mod = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
noise_snr = -50

mod_arr = np.empty([arr.shape[0],25])
snr_arr = np.empty([arr.shape[0],1])
for x in range(arr.shape[0]):
    mod_arr[x] = noise_mod
    snr_arr[x] = noise_snr
    
# Create a dataset "Y" with the random noise
# Create a dataset "Z" with SNR for the new noise datasets, 
# we will use -50 to represent noise
dsetY = f_new.create_dataset("Y", data=mod_arr)
dsetZ = f_new.create_dataset("Z", data=snr_arr)

del dsetY
del dsetZ
del samples

#Close the files  
f_new.close()
