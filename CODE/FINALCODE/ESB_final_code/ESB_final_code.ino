/*************************************************************
  Download latest Blynk library here:
    https://github.com/blynkkk/blynk-library/releases/latest

  Blynk is a platform with iOS and Android apps to control
  Arduino, Raspberry Pi and the likes over the Internet.
  You can easily build graphic interfaces for all your
  projects by simply dragging and dropping widgets.

    Downloads, docs, tutorials: http://www.blynk.cc
    Sketch generator:           http://examples.blynk.cc
    Blynk community:            http://community.blynk.cc
    Follow us:                  http://www.fb.com/blynkapp
                                http://twitter.com/blynk_app

  Blynk library is licensed under MIT license
  This example code is in public domain.

 *************************************************************
  This example shows how to use Arduino MKR 1010
  to connect your project to Blynk.

  Note: This requires WiFiNINA library
    from http://librarymanager/all#WiFiNINA

  Feel free to apply it to any other example. It's simple!
 *************************************************************/

/* Comment this out to disable prints and save space */


//http://help.blynk.cc/en/articles/2091699-keep-your-void-loop-clean

#define BLYNK_PRINT Serial

#include <SPI.h>
#include <WiFiNINA.h>
#include <BlynkSimpleWiFiNINA.h>
#include "file.h/file.h"

// You should get Auth Token in the Blynk App.
// Go to the Project Settings (nut icon).
char auth[] = "cVCe7ywCC7mAuw2flCsIYmhHlX3eSz3o";

// Your WiFi credentials.
// Set password to "" for open networks.
char ssid[] = SECRET_SSID;
char pass[] = SECRET_PASS;


int pin1 = A3;
int pin2 = A2;
int raw_emg[4];
int emg_sign[4];
String raw = "";
String emg = "";
int pred[4];

#define PIN1 V5

BlynkTimer timer;

void setup() {
  Serial.begin(9600);
  Blynk.begin(auth, ssid, pass);
  timer.setInterval(1000L, sendPredToBlynk);
}

void loop() {
 
  int i;  
   
  for(i = 0;i < 4;i++) {
    raw_emg[i] = analogRead(pin1);
    emg_sign[i] = analogRead(pin2);
  }

  raw = "";
  emg = "";
  
  for(i = 0;i < 4;i++) {
    raw += String(raw_emg[i]);
    emg += String(emg_sign[i]);
    
    if(i < 3) {
      raw+= ",";
      emg+= ",";
    }
     
  }
  
  Serial.print(raw + "," + emg);
  Serial.print("\n");

  delay(500);

  Blynk.run();
  timer.run();

  delay(50);

}

void sendPredToBlynk() {  
  int i;
  
  if(Serial.available()) {
    i = Serial.read(); 
    Blynk.virtualWrite(PIN1, i);  
  }
  delay(500);
}
