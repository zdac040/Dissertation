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


import serial

import time

import numpy as np

from numpy import argmax
from scipy.special import softmax

import tensorflow as tf
from tensorflow.keras.models import load_model
import struct


ser = serial.Serial('/dev/cu.usbmodem14101', 9600) 


array = []

model = load_model("model.h5")

for i in range(0, 30):
	v = (ser.readline()).decode('utf-8').rstrip("\n").split(",")
	v = [int(x,10) for x in v] 

	print(v)

	array = [v]

	print("")

	pred = np.array(array)
	print(pred)
	result = model.predict(pred)

	soft_prediction = tf.nn.softmax(result).numpy()
	print("")
	print("The softmax of prediction is: ", soft_prediction)

	class_pred = np.argmax(soft_prediction) #to obtain the predicted class
	print("")
	print("The class predicted is: ", class_pred)

	print("")

	ser.write(struct.pack('>B', class_pred))

ser.close()