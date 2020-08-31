import serial

import time

import numpy as np
import pandas as pd

from numpy import argmax
from scipy.special import softmax

import tensorflow as tf
from tensorflow.keras.models import load_model
import struct

from sklearn.model_selection import train_test_split

import xlsxwriter

ser = serial.Serial('/dev/cu.usbmodem14101', 9600) 


colNS = ['RAW', 'EMG', 'BINARY'] 
dataNS = pd.read_excel("FINALE/NULLSIGNAL.xlsx", names=colNS) 

# print(dataNS)

colMF = ['RAW', 'EMG', 'BINARY'] 
dataMF = pd.read_excel("FINALE/MIDDLEFINGER.xlsx", names=colMF) 

# print(dataMF)

colP = ['RAW', 'EMG', 'BINARY'] 
dataP = pd.read_excel("FINALE/PUNCH.xlsx", names=colP) 

# print(dataP)

colC = ['RAW', 'EMG', 'BINARY'] 
dataC = pd.read_excel("FINALE/CONTRACTION.xlsx", names=colC) 


num_rows = dataNS.shape[0]

trans_table_ns = np.zeros((num_rows-4+1,2*4+1)) 

for i in range(236):
  cur_slice = dataNS.iloc[i:i+4,:]
  raw_slice = np.array(cur_slice['RAW'])
  emg_slice = np.array(cur_slice['EMG'])
  classification = np.max(cur_slice['BINARY'])
  trans_table_ns[i,:] = np.append(np.concatenate((raw_slice,emg_slice)),classification)

# print(trans_table_ns)

num_rows = dataMF.shape[0]

trans_table_mf = np.zeros((num_rows-4+1,2*4+1)) 

for i in range(236):
  cur_slice = dataMF.iloc[i:i+4,:]
  raw_slice = np.array(cur_slice['RAW'])
  emg_slice = np.array(cur_slice['EMG'])
  classification = np.max(cur_slice['BINARY'])
  trans_table_mf[i,:] = np.append(np.concatenate((raw_slice,emg_slice)),classification)

# print(trans_table_mf)

num_rows = dataP.shape[0]

trans_table_p = np.zeros((num_rows-4+1,2*4+1)) 

for i in range(236):
  cur_slice = dataP.iloc[i:i+4,:]
  raw_slice = np.array(cur_slice['RAW'])
  emg_slice = np.array(cur_slice['EMG'])
  classification = np.max(cur_slice['BINARY'])
  trans_table_p[i,:] = np.append(np.concatenate((raw_slice,emg_slice)),classification)

# print(trans_table_p)

num_rows = dataC.shape[0]

trans_table_c = np.zeros((num_rows-4+1,2*4+1)) 

for i in range(236):
  cur_slice = dataC.iloc[i:i+4,:]
  raw_slice = np.array(cur_slice['RAW'])
  emg_slice = np.array(cur_slice['EMG'])
  classification = np.max(cur_slice['BINARY'])
  trans_table_c[i,:] = np.append(np.concatenate((raw_slice,emg_slice)),classification)

# print(trans_table_c)

xns = trans_table_ns[:,:-1]
yns = trans_table_ns[:,-1]

xmf = trans_table_mf[:,:-1]
ymf = trans_table_mf[:,-1]

xp = trans_table_p[:,:-1]
yp = trans_table_p[:,-1]

xc = trans_table_c[:,:-1]
yc = trans_table_c[:,-1]

xns = (xns-512)/1024
xmf = (xmf-512)/1024
xp = (xp-512)/1024
xc = (xc-512)/1024

x = np.concatenate((xns, xmf, xp, xc))

y = np.concatenate((yns, ymf, yp, yc))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

print(X_test)

model = load_model("model.h5")

for a in X_test:
	pred = np.array([a])
	print(pred)
	result = model.predict(pred)
	print(result)

	print("")

	soft_prediction = tf.nn.softmax(result).numpy()
	print("")
	print("The softmax of prediction is: ", soft_prediction)

	class_pred = np.argmax(soft_prediction) #to obtain the predicted class
	print("")
	print("The class predicted is: ", class_pred)

	ser.write(struct.pack('>B', class_pred))

	time.sleep(3)

ser.close()
