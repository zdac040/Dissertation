#https://towardsdatascience.com/how-to-drive-your-arduino-propulsed-robot-arm-with-python-4e428873237b
#https://www.quora.com/How-do-I-send-data-from-python-to-Arduino
#https://stackoverflow.com/questions/1514553/how-to-declare-an-array-in-python
#https://www.instructables.com/id/Sending-Data-From-Arduino-to-Python-Via-USB/
#https://stackoverflow.com/questions/24597929/how-to-convert-byte-to-int
#https://www.youtube.com/watch?v=yPbbZT9AQ7I
#https://www.it-swarm.dev/it/python/python-arduino-serial-read-write/1047267274/
#https://stackoverflow.com/questions/27639605/send-a-list-of-data-from-python-to-arduino-using-pyserial
#https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/#loading
#https://stackoverflow.com/questions/16643089/data-coming-from-arduino-to-python-raspberrypi-invalid-literal-error
#https://stackoverflow.com/questions/24074914/python-to-arduino-serial-read-write
#https://stackoverflow.com/questions/16641119/why-does-append-always-return-none-in-python
#https://create.arduino.cc/projecthub/officine-innesto/control-your-iot-cloud-kit-via-blynk-ec6a16
#https://www.pythonforbeginners.com/basics/list-comprehensions-in-python
#https://www.programiz.com/python-programming/methods/built-in/int
#https://community.blynk.cc/t/led-widget-controlled-by-a-button-widget/24505/3
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
#https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
#https://machinelearningmastery.com/make-predictions-scikit-learn/
#https://www.machinecurve.com/index.php/2020/02/21/how-to-predict-new-samples-with-your-keras-model/#generating-predictions
#https://stackoverflow.com/questions/29899377/indexerror-list-index-out-of-range-in-array-search
#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
#https://stackoverflow.com/questions/16789776/iterating-over-two-values-of-a-list-at-a-time-in-python
#https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
#https://stackoverflow.com/questions/31909722/how-to-write-python-array-into-excel-spread-sheet

import serial

import time

import numpy as np
import pandas as pd

from numpy import argmax
from scipy.special import softmax

import tensorflow as tf
from tensorflow.keras.models import load_model
import struct

import termplot
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

mean, sigma = 0, 450
noise = np.random.normal(mean, sigma, [8]) 
# print("Noise is: ", noise)

# print("")

xns = trans_table_ns[:,:-1]
yns = trans_table_ns[:,-1]

signal_noise_ns = xns + noise
print("Signal is: ", signal_noise_ns)

xmf = trans_table_mf[:,:-1]
ymf = trans_table_mf[:,-1]

signal_noise_mf = xmf + noise
print("Signal is: ", signal_noise_mf)

xp = trans_table_p[:,:-1]
yp = trans_table_p[:,-1]

signal_noise_p = xp + noise
print("Signal is: ", signal_noise_p)

xc = trans_table_c[:,:-1]
yc = trans_table_c[:,-1]

signal_noise_c = xc + noise
print("Signal is: ", signal_noise_c)

#save the data into relative excel files

# workbookNS = xlsxwriter.Workbook('attackNS.xlsx')
# worksheetNS = workbookNS.add_worksheet()

# row = 0
# for col, data in enumerate(signal_noise_ns):
# 	worksheetNS.write_column(row, col, data)

# workbookNS.close()

# workbookMF = xlsxwriter.Workbook('attackMF.xlsx')
# worksheetMF = workbookMF.add_worksheet()

# row = 0
# for col, data in enumerate(signal_noise_mf):
# 	worksheetMF.write_column(row, col, data)

# workbookMF.close()

# workbookP = xlsxwriter.Workbook('attackP.xlsx')
# worksheetP = workbookP.add_worksheet()

# row = 0
# for col, data in enumerate(signal_noise_p):
# 	worksheetP.write_column(row, col, data)

# workbookP.close()

# workbookC = xlsxwriter.Workbook('attackC.xlsx')
# worksheetC = workbookC.add_worksheet()

# row = 0
# for col, data in enumerate(signal_noise_c):
# 	worksheetC.write_column(row, col, data)

# workbookC.close()

signal_noise_ns = (signal_noise_ns-512)/1024
signal_noise_mf = (signal_noise_mf-512)/1024
signal_noise_p = (signal_noise_p-512)/1024
signal_noise_c = (signal_noise_c-512)/1024

x = np.concatenate((signal_noise_ns, signal_noise_mf, signal_noise_p, signal_noise_c))

y = np.concatenate((yns, ymf, yp, yc))


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)

model = load_model("model.h5")

for a in X_test:
  pred = np.array([a])
  print(pred)
  result = model.predict(pred)
  print(result)

  print("")

  soft_prediction = tf.nn.softmax(result).numpy()
  print("The softmax of prediction is: ", soft_prediction)

  print("")

  class_pred = np.argmax(soft_prediction) #to obtain the predicted class
  print("The class predicted is: ", class_pred)

  ser.write(struct.pack('>B', class_pred))

  print("")

  time.sleep(2)

ser.close()





