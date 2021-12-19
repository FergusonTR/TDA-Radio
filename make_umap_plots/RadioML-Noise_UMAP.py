import numpy as np
import h5py
import pandas as pd
import time
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
#import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#f = h5py.File('c:/tmp/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
#f = h5py.File('c:/tmp/2018.01/noise_capture_20211123.hdf5','r')
f = h5py.File('/home/terry/2018.01/noise_capture_20211123.hdf5','r')
#f = h5py.File('/home/terry/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5','r')
#f = h5py.File('/home/terry/2018.01/ExtractDataset/mod_4ASK.h5','r')
print("Opened File.....")

classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK',
           '32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM',
           '256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
           'GMSK','OQPSK','Noise']

#load data into np arrays
f1 = f['X'][:] #Array of I-Q data
f2 = f['Y'][:] #Array of Modulation modes
f3 = f['Z'][:] #Array of SNR data

f.close()

perc = float(input("Enter percent of data (.1 to 1):  "))
neigh = int(input("Enter the number of nearest neighbors: "))

modulation = []

for entry in range(len(f2)):
    x = np.where(f2[entry] == 1)
    x = x[0][0]
    modulation.append(classes[x])   

#merge data
f1 = f1.reshape(f1.shape[0],f1.shape[1]*f1.shape[2]) #reshape the dataset into a vector array
#f1 = f1.reshape(2555904,2048) #reshape the dataset into a vector array
f4 = np.hstack((f3, f1))
print("Merging complete......")

#Delete unecessary arrays  
del f1, f2, f3
print("Deleted Array's.....")

#Column Names
column_names =['SNR']
for IQ_no in range(1024):
    I_col = 'I'+ str(IQ_no)
    Q_col = 'Q'+ str(IQ_no)
    column_names.append(I_col)
    column_names.append(Q_col)
print("Column Array is complete......")

#create dataframe
RadioML = pd.DataFrame(data=f4,columns=column_names)
RadioML['modulation'] = modulation

del f4, modulation, column_names

RadioML = RadioML.sample(frac=perc, random_state=42)
RadioML_Data = RadioML.iloc[:, 1:-1]
#RadioML_Data = RadioML
print("DataFrame Generated.....")

print("Starting UMAP process......")
ts = time.time()
reducer = umap.UMAP(n_neighbors=neigh, metric='euclidean', low_memory=True)
scaled_RadioML_Data = StandardScaler().fit_transform(RadioML_Data)
embedding =reducer.fit_transform(scaled_RadioML_Data)
filename = 'this_is_a_test.sav'
joblib.dump(embedding, filename)
ts = time.time() - ts
print("Embedding complete....",ts)

#setup labels
color_labels = RadioML['SNR'].unique()
color_labels.sort()

fig, ax = plt.subplots(1, figsize=(14, 10))

im = plt.scatter(
    embedding[:, 0],
    embedding[:, 1], 
    c=RadioML['SNR'],   
    cmap='Spectral',
    s=0.3,
    alpha=1.0
    )

plt.setp(ax, xticks=[], yticks=[])
plt.colorbar(im)
plt.title('Radio ML 2018 Dataset UMAP')
plt.show()

del  embedding, RadioML, RadioML_Data




