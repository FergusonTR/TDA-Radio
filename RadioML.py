import numpy as np
import h5py
import matplotlib.pyplot as plt
from ripser import Rips
from persim import PersImage
from persim import PersistenceImager

f = h5py.File('c:/tmp/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')

xdset = f['X'] #'I-Q data points'
ydset = f['Y'] #'Sequence number'
zdset = f['Z'] #'Noise Floor'

seq = 246577

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
    
#for x in range(1024):
#    plt.scatter(x_seq[x,0],x_seq[x,1],s=10)
    
plt.scatter(x_seq[:,0],x_seq[:,1],s=5)

title_string = classes[y_label],'  ',z_seq[0]

plt.title(title_string, fontsize=14)

plt.ylabel("In-Phase",fontsize =10)
plt.xlabel("Quadrature",fontsize=10)

plt.show()

rips = Rips(maxdim=1, coeff=2)
diagrams = [rips.fit_transform(x_seq)]
diagrams_h1 = [rips.fit_transform(x_seq)[1]]

plt.figure(figsize=(12,6))
plt.subplot(121)

rips.plot(diagrams[0], show=False)
plt.title("PD of $H_1$")

rips.plot(diagrams_h1[-1], show=False)
plt.title("PD of $H_1$")

plt.show()

pimgr = PersistenceImager(pixel_size=100)
pimgr.fit(diagrams_h1)
imgs = pimgr.transform(diagrams_h1)

pimgr

imgs_array = np.array([img.flatten() for img in imgs])

plt.figure(figsize=(15,7.5))

for i in range(1):
    ax = plt.subplot(240+i+1)
    pimgr.plot_image(imgs[i], ax)
    plt.title("PI of $H_1$ for noise")

for i in range(1):
    ax = plt.subplot(240+i+5)
    pimgr.plot_image(imgs[-(i+1)], ax)
    plt.title("PI of $H_1$ for circle w/ noise")