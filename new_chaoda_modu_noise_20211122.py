#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 21:09:48 2021

@author: Terry Ferguson
"""
import random
import numpy as np
import h5py
import pandas as pd
from time import time
import datetime
from pyclam import CHAODA
import numpy
from sklearn.metrics import roc_auc_score
import meta_models
from typing import Callable
from typing import List
from typing import Tuple
import os.path
import csv

# modulation classes from the RadioML 2018 + noise
classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK',
           '32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM',
           '256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM',
           'GMSK','OQPSK','Noise']

dir_path = r'/home/terry/2018.01/ExtractDataset'

#Open the generated noise file
f_noise = h5py.File('/home/terry/2018.01/noise_capture_20211123.hdf5','r')
f1_noise = f_noise['X'][0:36864] #Array of I-Q data (noise)
f2_noise = f_noise['Y'][0:36864] #Array of Modulation modes (noise)
f3_noise = f_noise['Z'][0:36864] #Array of SNR data (noise)

# Open the RadioML 2018 dataset
for modu in range(24):
    filename = dir_path + '/mod_' + classes[modu]+ '.h5'
    f = h5py.File(filename,'r')
    
    print("Starting ",classes[modu])
       
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S Opened File(s)....."))
   
    for snr in range(26):
       
        # reload data into np arrays
        f1 = f['X'][:] #Array of I-Q data
        f2 = f['Y'][:] #Array of Modulation modes
        f3 = f['Z'][:] #Array of SNR data
    
        # Adding an extra column to match the new "noise" column added
        extra_col = np.full((f1.shape[0],1),0)
        f2 = np.hstack((f2,extra_col))
        
        snr_val = ((snr+1) * 2) - 22
        
        print("Modulation: ",classes[modu]," SNR ", snr_val,"dBm")
                
        #reduce the set to the noise every 2dB
        snr_start = (snr * 4096)
        snr_end = snr_start + 4096 - 1
        f1_SNR = f1[snr_start:snr_end]
        f2_SNR = f2[snr_start:snr_end]
        f3_SNR = f3[snr_start:snr_end]    
    
        # combine the noise data and the RadioML data
        f1 = np.concatenate((f1_SNR,f1_noise), axis=0)
        f2 = np.concatenate((f2_SNR,f2_noise), axis=0)
        f3 = np.concatenate((f3_SNR,f3_noise), axis=0)
        
        # Close the dataset to conserve memory
        #f.close()
            
        # Clean up arrays to perserve memory
        del extra_col
        
        # Set Arrays for holding 
        modulation = []
        signal = []
        
        # Generate an array of modulation modes
        for entry in range(len(f2)):
            x = np.where(f2[entry] == 1)
            x = x[0][0]
        #   modulation.append(classes[x])
            modulation.append(x)
            
        # Added for noise labels != -50 dBM
        for entry in range(len(f3)):
            if f3[entry][0] != -50:
                signal.append(1)
            else:
                signal.append(0)       
        
        # merge data
        f1 = f1.reshape(f1.shape[0],f1.shape[1]*f1.shape[2]) #reshape the dataset into a vector array
        f4 = np.hstack((f3, f1))
        
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S Merging complete....."))
        
        # Delete unecessary arrays  
        del f1, f2, f3
        
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S Deleted Array's....."))
        
        # Column Names
        column_names =['SNR']
        for IQ_no in range(1024):
            I_col = 'I'+ str(IQ_no)
            Q_col = 'Q'+ str(IQ_no)
            column_names.append(I_col)
            column_names.append(Q_col)
            
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S Column Array is complete....."))    
        
        # create dataframe
        RadioML = pd.DataFrame(data=f4,columns=column_names)
        RadioML['modulation'] = modulation
        RadioML['signal'] = signal
        
        del f4, modulation, column_names, signal
        
        # Remove the labels and determine if you are going to sample
        # RadioML = RadioML.sample(frac=1, random_state=42)
        RadioML_Data = RadioML.iloc[:, 1:-2]
        data = RadioML_Data.to_numpy()
        labels = RadioML[['signal']]
        labels = labels.to_numpy()
        
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S DataFrame Generated.....\n"))
        
        np.random.seed(42), random.seed(42)
        
        # Defaults from original CHAODA benchmark code
        #-----------------------------------------------------------------
        META_MODELS: List[Tuple[str, str, Callable[[numpy.array], float]]] = [
            # tuple of (metric, method, function)
            (name.split('_')[1], '_'.join(name.split('_')[2:]), method)
            for name, method in meta_models.META_MODELS.items()
        ]
        
        METRICS = ['cityblock', 'euclidean']
        #NORMALIZE = 'gaussian'
        SUB_SAMPLE = 64_000  # for testing the implementation
        MAX_DEPTH = 50  # even though no dataset reaches this far
        fast = True
        min_points = max((data.shape[0] // 1000), 1)
        speed_threshold = max(128, int(np.sqrt(len(labels)))) if fast else None
        print(f'speed threshold set to {speed_threshold}')
        
        csv_path ='/home/terry/TDA-Radio/chaoda-master/results/individual_scores.csv'
        scores_path ='/home/terry/TDA-Radio/chaoda-master/results/scores.csv'
        
        start = time()
        detector: CHAODA = CHAODA(
            metrics=METRICS,
            max_depth=MAX_DEPTH,
            min_points=min_points,
            meta_ml_functions=META_MODELS,
            speed_threshold=speed_threshold,
        ).fit(data=data)
        
        if csv_path is not None:
            # Print individual method scores.   
            
            if not os.path.exists(csv_path):    
              with open(csv_path, 'w') as individuals_csv:    
                        columns = ','.join([
                            'dataset',
                            'cardinality',
                            'dimensionality',
                            'num_components',
                            'num_clusters',
                            'num_edges',
                            'min_depth',
                            'max_depth',
                            'method',
                            'auc_roc',     
                        ])
                        individuals_csv.write(f'{columns}\n')
                        
        if scores_path is not None:  
            if not os.path.exists(scores_path):    
              with open(scores_path, 'w') as scores_csv:    
                        columns = ','.join([
                            'dataset',
                            'roc_score',
                            'time'                               
                        ])
                        scores_csv.write(f'{columns}\n')
        
        index = 0
        # noinspection PyProtectedMember
        for method_name, graph in detector._graphs:
            # noinspection PyProtectedMember
            scores = detector._individual_scores[index]
            auc = roc_auc_score(labels, scores)
            index += 1
        
            data_name = classes[modu]+'_'+str(snr_val)
            
            with open(csv_path, 'a') as individuals_csv:
                features = ','.join([
                    data_name,
                    f'{data.shape[0]}',
                    f'{data.shape[1]}',
                    f'{len(graph.components)}',
                    f'{graph.cardinality}',
                    f'{len(graph.edges)}',
                    f'{graph.depth_range[0]}',
                    f'{graph.depth_range[1]}',
                    method_name,
                    f'{auc:.2f}',
                ])
                individuals_csv.write(f'{features}\n')
                
        
        group_score = roc_auc_score(labels,detector.scores)
        group_time = time() - start
                
        with open(scores_path, 'a') as scores_csv:
                features = ','.join([
                    data_name,
                    f'{group_score}',
                    f'{group_time}',
                ])
                scores_csv.write(f'{features}\n')
                
        
            
            



