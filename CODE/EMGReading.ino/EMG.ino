//https://github.com/sandroormeno/TUTORIAL-TENSORFLOW-LITE-ARDUINO-DUE/blob/master/arduino/exercice_final/exercice_final.ino

//https://forum.arduino.cc/index.php?topic=612565.0

int pin1 = A3;
int pin2 = A2;
int raw_emg = 0;
int emg_sign = 0;
int stimulus = 0;

void setup() {
  
  Serial.begin(9600);
    
}

void loop() {
  
  raw_emg = analogRead(pin1);
  emg_sign = analogRead(pin2);

   // send data only when you receive data:
  if (Serial.available() > 0) {
      Serial.read();
      stimulus = 1;
  }
    // analog reads
    Serial.print(raw_emg);
    Serial.print("\t");
    Serial.print(emg_sign);
    Serial.print("\t");
    Serial.println(stimulus);
    stimulus = 0;
    
  delay(500);

}
