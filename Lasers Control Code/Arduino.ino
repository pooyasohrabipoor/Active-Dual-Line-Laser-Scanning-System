int pwmpin=9;
int dutycycle=45 ;/// 0-100


void setup() {
 
  // put your setup code here, to run once:
pinMode(pwmpin,OUTPUT);
analogWrite(pwmpin,dutycycle);
Serial.begin(9600);
//delay(1000);

Serial.write('A');


}

void loop() {
  
  //Serial.write('A');

  // put your main code here, to run repeatedly:
  if (Serial.available()>0){
     char signal=Serial.read();
   
    if (signal=='C'){     ///   it is C
      Serial.println(signal);
    
       dutycycle=dutycycle+1;
      analogWrite(pwmpin,dutycycle);
      delay(100);
      Serial.write('A'); //// finished dutycyle now do image processing
    }

  }

}
