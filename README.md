# EMGdissertation
This repository contains the practical part of my MSc final year project in Information Security. It focuses on the use of Electromyography(EMG) and the security behind it. 

I built a protoype, composed by Arduino mkr1010 connected to the Myoware sensor through the use of three RGB cables, inserted in the right ports. The prototype is connected to the Arduino software code via USB cable. The user places the electrodes of the Myoware sensor on the forearm. The program on the Arduino software is uploaded and the user can start contracting the arm. The registered signals are visible on the serial monitor window of the Arduino software, accessible at the top right corner of the program. The Arduino software code specifies to send four signals at a time to a python code through the use of a serial port, Pyserial. Python code collects the signals at a time and inputs them in a saved Artificial Neural Network, created on tensorflow, which outputs the prediction. The latter is sent back to the Arduino software code to control the volume of Blynk, an IoT device of my choice.

For the project realisation, I have used Arduino software, Sublime text editor and Tensorflow software:
In this repository are present four files:
1. The main code that was developed on the Arduino software, which uses language c++. In it, it is possible to find:
  - the piece of code for recording the EMG and the raw data
  - the piece of code line to send the recorded signals to the relative python code, through the use of Pyserial, for the relative pattern recognition     processing. 
  - the piece of code to get back the prediction of the various gestures calculated in file 3. 
  - the piece of code to control the volume of teh Gauge button created on Blynk.

2. The Tensorflow code where the Artificial Neural Network has been developed. 

3. The python code, was developed in Sublime text and contains:
  - the piece of code to receive both the EMG and raw signals and prepare them for the classification. 
  - the line to load the saved model that calculates the prediction.
  - the line of code to send the prediction to the Arduino software code.
    
4. This file is a copy of the file 3. but with the addition of the attack code that appears as a simplified version of a adversarial attack. I have        input some white Gaussian noise in the signal to change the prediction, before sending it back to the Arduino software code.
