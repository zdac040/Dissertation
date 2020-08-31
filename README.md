# EMGdissertation
This repository contains the practical part of my MSc final year project in Information Security. It focuses on the use of Electromyography(EMG) and the security behind it. 

The project is structured in the following way:
1. In the folder DatasetsforTheProject there are the datasets I have used to train the Tensorflow model on Google Colab and to perform predictions and the attack.
2. The folder CODE contains:
 - EMGReading.ino which is the Arduino software file that I coded initially to read raw, EMG and binary. I have added it in case someone wants to test it, but is not needed for the final code.
 - FINALCODE contains the final project. I set it up as so:
   - ESB_final_code.ino and ESB_final_code.py are respectively the Arduino software code and the python code. Final_attack.py is the file of the attack.
   - model.h5 is the saved model that is already uploaded in all python codes.
   - TensorFlowModel contains the Google Colab neural network (this is not needed when running the project since in it I created the neural network).
   - DatasetsPyserial+Attack contains the original file and the attack one but updated to be used with the datasets.
   - file.h contains the file to insert the Wifi and password.
 

In order to execute the programs:

USING THE PROTOTYPE

- Download Blynk app and create a button
- Update the Blynk authentication code in the ESB_final_code.ino with your new one.
- Insert in file.h your Wifi credentials.
- Take the the prototype and connect it to the computer via USB cable. 
- Open ESB_final_code.ino in Arduino software and load the file.
- Open ESB_final_code.py if you want to predict only or Final_attack.py if you want to run the attack (The two python files are run individually)
- Open the terminal and go in the directoy of the project and run python ESB_final_code.py or python Final_attack.py

USING THE DATASETS

The process is the same of when using the prototype but the files run are:
- python attack_w:out_noise.py or python final_attack_loadingData.py

It is important that the file.h and the model.h5 are in the same folder with the Arduino code and the python code that you want to run in order to be executed.

Also, in the files attack_w:out_noise.py and final_attack_loadingData.py I have uploaded the datasets (taking below as an example the null signal gesture) in the following way: 

dataNS = pd.read_excel("FINALE/NULLSIGNAL.xlsx", names=colNS)

"FINALE/NULLSIGNAL.xlsx" needs to be changed in DatasetsforTheProject/NULLSIGNAL.xlsx since now all datasets are in the folder DatasetsforTheProject

For the project realisation, I have used Arduino software, Sublime text editor and Tensorflow software.
