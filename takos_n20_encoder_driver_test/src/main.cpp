#include <Arduino.h>
#include <driver/gpio.h>
#include <esp_arduino_version.h>

constexpr uint32_t PWM_HZ = 20000;
constexpr uint32_t REVERSE_COAST_MS = 200;

constexpr float GEAR_RATIO = 380.0f; // Motor revolutions per output revolution.
// REQUIRED: full cycles on ONE encoder channel per MOTOR-shaft revolution.
// These are not x4 edge counts. The provided motor specs do not include PPR.
// Set each value from the encoder datasheet; zero means unknown.
constexpr float MOTOR_1_ENCODER_PPR = 0.0f;
constexpr float MOTOR_2_ENCODER_PPR = 0.0f;
constexpr float QUADRATURE_MULTIPLIER = 4.0f;

struct Motor {
  uint8_t in1, in2, channel1, channel2;
  gpio_num_t encoderA, encoderB;
  float countsPerOutputRev; // x4 gearbox-output counts/rev; 0 disables RPM.
  int duty = 160;
  int direction = 0; // Requested direction; zero means stopped.
  bool reversing = false;
  uint32_t reverseStartedMs = 0;
  volatile int64_t count = 0;
  volatile uint8_t previousAB = 0;
  int64_t lastReportCount = 0;
  uint32_t lastReportMs = 0;

  Motor(uint8_t p1, uint8_t p2, uint8_t c1, uint8_t c2,
        gpio_num_t a, gpio_num_t b, float countsPerRev)
      : in1(p1), in2(p2), channel1(c1), channel2(c2),
        encoderA(a), encoderB(b), countsPerOutputRev(countsPerRev) {}
};

// ISR counts all four quadrature edges per encoder cycle.
Motor motors[] = {
    {25, 26, 0, 1, GPIO_NUM_32, GPIO_NUM_35,
     MOTOR_1_ENCODER_PPR * QUADRATURE_MULTIPLIER * GEAR_RATIO},
    {14, 27, 2, 3, GPIO_NUM_13, GPIO_NUM_33,
     MOTOR_2_ENCODER_PPR * QUADRATURE_MULTIPLIER * GEAR_RATIO},
};
constexpr uint8_t MOTOR_COUNT = 2;
uint8_t selectedMask = 0x03; // Start with both motors selected.
portMUX_TYPE encoderMux = portMUX_INITIALIZER_UNLOCKED;
DRAM_ATTR const int8_t QUADRATURE[16] = {
    0, -1, 1, 0, 1, 0, 0, -1,
    -1, 0, 0, 1, 0, 1, -1, 0};

uint8_t IRAM_ATTR readAB(const Motor &motor) {
  return (gpio_get_level(motor.encoderA) << 1) | gpio_get_level(motor.encoderB);
}

void IRAM_ATTR updateEncoder(Motor &motor) {
  portENTER_CRITICAL_ISR(&encoderMux);
  const uint8_t current = readAB(motor);
  motor.count += QUADRATURE[(motor.previousAB << 2) | current];
  motor.previousAB = current;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void IRAM_ATTR encoder1ISR() { updateEncoder(motors[0]); }
void IRAM_ATTR encoder2ISR() { updateEncoder(motors[1]); }

int64_t readCount(Motor &motor) {
  portENTER_CRITICAL(&encoderMux);
  const int64_t count = motor.count;
  portEXIT_CRITICAL(&encoderMux);
  return count;
}

void writePWM(uint8_t pin, uint8_t channel, uint8_t value) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, value);
#else
  ledcWrite(channel, value);
#endif
}

void coast(Motor &motor) {
  writePWM(motor.in1, motor.channel1, 0);
  writePWM(motor.in2, motor.channel2, 0);
}

void applyDrive(Motor &motor) {
  coast(motor);
  if (motor.reversing) return;
  if (motor.direction > 0) writePWM(motor.in1, motor.channel1, motor.duty);
  if (motor.direction < 0) writePWM(motor.in2, motor.channel2, motor.duty);
}

void startMotor(Motor &motor, int requestedDirection) {
  if (motor.direction != 0 && motor.direction != requestedDirection) {
    coast(motor);
    motor.reversing = true;
    motor.reverseStartedMs = millis();
  }
  motor.direction = requestedDirection;
  applyDrive(motor);
}

void stopMotor(Motor &motor) {
  motor.direction = 0;
  motor.reversing = false;
  coast(motor);
}

void printSelection() {
  Serial.printf("Selected: %s\n", selectedMask == 3 ? "both motors" :
                selectedMask == 1 ? "motor 1" : "motor 2");
}

void printHelp() {
  Serial.println("\nDual N20 / DRV8833 test, 115200 baud. Both start stopped.");
  Serial.println("1: select motor 1 | 2: select motor 2 | b: select both");
  Serial.println("Selected motors: f: forward | r: reverse | s: stop/coast");
  Serial.println("+: duty +15 | -: duty -15 | z: zero selected encoder(s)");
  Serial.println("!: stop BOTH immediately | h: help");
  Serial.println("f/r run indefinitely until stopped. Duty 0..255, initially 160.");
  Serial.println("Example: 1f2r runs motor 1 forward and motor 2 reverse.");
  Serial.println("Output RPM uses 380:1 gearing and each motor's encoder PPR.");
  Serial.println("3 / 7: set selected encoder(s) to 3 / 7 PPR and enable RPM.");
  Serial.println("PPR selection is temporary; configure source constants to keep it.");
  printSelection();
}

void setup() {
  // Keep all four driver inputs low before attaching PWM.
  for (Motor &motor : motors) {
    pinMode(motor.in1, OUTPUT);
    pinMode(motor.in2, OUTPUT);
    digitalWrite(motor.in1, LOW);
    digitalWrite(motor.in2, LOW);
  }
  Serial.begin(115200);
  bool pwmOK = true;
  for (Motor &motor : motors) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    const bool ok1 = ledcAttach(motor.in1, PWM_HZ, 8);
    const bool ok2 = ledcAttach(motor.in2, PWM_HZ, 8);
    pwmOK = pwmOK && ok1 && ok2;
#else
    const double hz1 = ledcSetup(motor.channel1, PWM_HZ, 8);
    const double hz2 = ledcSetup(motor.channel2, PWM_HZ, 8);
    pwmOK = pwmOK && hz1 > 0 && hz2 > 0;
    ledcAttachPin(motor.in1, motor.channel1);
    ledcAttachPin(motor.in2, motor.channel2);
#endif
    coast(motor);
  }
  if (!pwmOK) {
    for (Motor &motor : motors) coast(motor);
    Serial.println("PWM initialization failed; test halted.");
    while (true) delay(1000);
  }

  // Signals must be 3.3 V compatible. GPIO35 has no internal pull-up.
  for (Motor &motor : motors) {
    pinMode(motor.encoderA, INPUT);
    pinMode(motor.encoderB, INPUT);
    motor.previousAB = readAB(motor);
    motor.lastReportMs = millis();
  }
  attachInterrupt(digitalPinToInterrupt(motors[0].encoderA), encoder1ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(motors[0].encoderB), encoder1ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(motors[1].encoderA), encoder2ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(motors[1].encoderB), encoder2ISR, CHANGE);
  printHelp();
}

void handleCommand(char command) {
  if (command >= 'A' && command <= 'Z') command += 'a' - 'A';
  switch (command) {
  case '1': selectedMask = 1; printSelection(); return;
  case '2': selectedMask = 2; printSelection(); return;
  case 'b': selectedMask = 3; printSelection(); return;
  case '!':
    for (Motor &motor : motors) stopMotor(motor);
    Serial.println("Both motors stopped (coast).");
    return;
  case 'h': case '?': printHelp(); return;
  case '\r': case '\n': case ' ': return;
  case 'f': case 'r': case 's': case '+': case '-': case 'z':
  case '3': case '7': break;
  default: Serial.println("Unknown command; press h for help."); return;
  }
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
    if (!(selectedMask & (1 << i))) continue;
    Motor &motor = motors[i];
    switch (command) {
    case '3': case '7':
      motor.countsPerOutputRev = (command - '0') * QUADRATURE_MULTIPLIER * GEAR_RATIO;
      Serial.printf("M%u encoder PPR=%d; counts/output-rev=%.0f\n",
                    i + 1, command - '0', motor.countsPerOutputRev);
      break;
    case 'f': startMotor(motor, 1); break;
    case 'r': startMotor(motor, -1); break;
    case 's': stopMotor(motor); break;
    case '+': motor.duty = min(255, motor.duty + 15); applyDrive(motor); break;
    case '-': motor.duty = max(0, motor.duty - 15); applyDrive(motor); break;
    case 'z':
      portENTER_CRITICAL(&encoderMux);
      motor.count = 0;
      portEXIT_CRITICAL(&encoderMux);
      motor.lastReportCount = 0;
      motor.lastReportMs = millis();
      Serial.printf("M%u count zeroed.\n", i + 1);
      break;
    }
  }
}

void loop() {
  if (Serial.available()) handleCommand(Serial.read());
  const uint32_t now = millis();
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
    Motor &motor = motors[i];
    if (motor.reversing && now - motor.reverseStartedMs >= REVERSE_COAST_MS) {
      motor.reversing = false;
      applyDrive(motor);
    }
    const uint32_t elapsed = now - motor.lastReportMs;
    if (elapsed < 500) continue;
    const int64_t count = readCount(motor);
    const double rate = (count - motor.lastReportCount) * 1000.0 / elapsed;
    const char *state = motor.reversing ? "reversal-coast" :
                        motor.direction > 0 ? "forward" :
                        motor.direction < 0 ? "reverse" : "coast";
    Serial.printf("M%u drive=%s duty=%d/255 count=%lld counts/s=%.1f A=%d B=%d",
                  i + 1, state, motor.duty, static_cast<long long>(count), rate,
                  digitalRead(motor.encoderA), digitalRead(motor.encoderB));
    if (motor.countsPerOutputRev > 0.0f) {
      const double signedRPM = rate * 60.0 / motor.countsPerOutputRev;
      Serial.printf(" encoder_PPR=%.0f output_RPM=%.2f",
                    motor.countsPerOutputRev / (QUADRATURE_MULTIPLIER * GEAR_RATIO),
                    signedRPM < 0 ? -signedRPM : signedRPM);
    } else {
      Serial.print(" output_RPM=N/A(send_3_or_7_to_set_PPR)");
    }
    Serial.println();
    motor.lastReportCount = count;
    motor.lastReportMs = now;
  }
}
