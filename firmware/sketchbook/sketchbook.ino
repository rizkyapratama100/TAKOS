#define stepX 25
#define dirX  26

#define stepY 27
#define dirY  14

#define stepZ 12
#define dirZ  13

#define enPin 4

void setup() {
  pinMode(stepX, OUTPUT);
  pinMode(dirX,  OUTPUT);
  pinMode(stepY, OUTPUT);
  pinMode(dirY,  OUTPUT);
  pinMode(stepZ, OUTPUT);
  pinMode(dirZ,  OUTPUT);
  pinMode(enPin, OUTPUT);

  digitalWrite(enPin, LOW);
  digitalWrite(dirX, HIGH);
  digitalWrite(dirY, LOW);
  digitalWrite(dirZ, HIGH);
}

void loop() {
  for(int x = 0; x < 800; x++) {
    digitalWrite(stepX, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepX, LOW);
    delayMicroseconds(1000);
  }
  delay(1000);

  for(int x = 0; x < 800; x++) {
    digitalWrite(stepY, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepY, LOW);
    delayMicroseconds(1000);
  }
  delay(1000);
}