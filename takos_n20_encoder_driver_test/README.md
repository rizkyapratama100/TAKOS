# Dual N20 motor and encoder test

ESP32 DOIT DevKit V1 with Arduino and DRV8833. No additional libraries required.

| Connection | Motor 1 | Motor 2 |
| --- | --- | --- |
| ESP32 to driver inputs | GPIO25 -> IN1, GPIO26 -> IN2 | GPIO14 -> IN3, GPIO27 -> IN4 |
| Driver outputs to motor | OUT1 -> M1, OUT2 -> M2 | OUT3 -> M1, OUT4 -> M2 |
| Encoder inputs to ESP32 | C1 -> GPIO32, C2 -> GPIO35 | C1 -> GPIO13, C2 -> GPIO33 |

Connect ESP32, driver, both encoders, and motor-supply grounds together. Power
VM from a supply suitable for both motors and the driver. Do not power motors
from the ESP32 3.3 V output. If the board exposes nSLEEP/SLP, it must be high;
check whether the breakout already pulls it high. Each input/output pair must
belong to the same H-bridge.

Power each encoder according to its specification. C1/C2 must deliver 3.3 V
compatible signals; use level shifting for 5 V push-pull outputs. Open-collector
outputs need external pull-ups, typically 4.7-10 kOhm to 3.3 V. All encoder pins
are configured as plain inputs; GPIO35 has no internal pull-up.

## Run and control

Build and upload with PlatformIO. Open Serial Monitor at 115200 baud. Secure
both motors with their shafts free. Both start stopped, with both selected.

Commands are single characters; no line ending is required. Selection persists
until changed. Selecting a motor does not change either motor's running state.

| Command | Action |
| --- | --- |
| `1` | Select motor 1 |
| `2` | Select motor 2 |
| `b` | Select both motors (default) |
| `f` / `r` | Run selected motors forward / reverse indefinitely |
| `s` | Stop selected motors (coast) |
| `!` | Stop both motors regardless of selection |
| `+` / `-` | Change selected motors' duty by 15, within 0-255 |
| `z` | Zero selected encoder counts |
| `3` / `7` | Set selected encoders to 3 / 7 PPR and enable RPM |
| `h` | Print help |

Examples:

- `1f2r`: motor 1 forward, motor 2 reverse.
- `1+`: increase motor 1 duty while motor 2 continues unchanged.
- `2s`: stop motor 2 while motor 1 continues unchanged.
- `bf`: run both forward at their respective duty settings.
- `bs` or `!`: stop both.

Each motor starts at duty 160/255 and keeps its own duty and direction. There
is no run timeout. Reversal includes 200 ms of coasting without blocking serial
commands or the other motor. `s`/`!` cancel pending reversal. Coasting does not
stop a shaft instantly. This open-loop test does not regulate speed to 40 RPM.

Each motor prints direction, duty, signed count, counts/second, and instantaneous
A/B levels every 500 ms, labeled M1/M2. Both edges of both encoder channels are
counted (x4 quadrature); invalid two-bit transitions are ignored. Count direction
should reverse when rotation reverses; its sign relative to `f` depends on wiring.

## Output-shaft RPM

The configured gearbox ratio is 380:1. Set `MOTOR_1_ENCODER_PPR` and
`MOTOR_2_ENCODER_PPR` near the top of `src/main.cpp` to the number of full cycles
on **one encoder channel per motor-shaft revolution**. The firmware calculates:

`counts per output revolution = encoder PPR * 4 * 380`

`output RPM = abs(count change) * 60000 / (elapsed milliseconds * counts per output revolution)`

Speed prints as nonnegative `output_RPM` every 500 ms; signed counts and
counts/second retain encoder direction. The drive label reports commanded
direction. Each RPM reading is an average over the reporting interval, so a
stop can take one interval to show zero after the shaft stops physically.

PPR is currently unknown and defaults to zero, displaying
`output_RPM=N/A(send_3_or_7_to_set_PPR)` rather than an invented speed. The supplied
6 V, 40 RPM, 380:1 specifications do not include encoder resolution. The 40 RPM
rating is a nominal operating specification, not a measurement or calibration.
Obtain encoder PPR from the supplier. If its specification is already x4 counts
per motor revolution, divide it by four before entering it as PPR. If it gives
counts per output revolution, divide that by `4 * 380` instead.

You can test either candidate resolution using serial commands: `b3f` sets both
encoders to 3 PPR and runs both motors forward; `b7` switches both to 7 PPR.
For separate settings, use `13` for motor 1 at 3 PPR and `27` for motor 2 at
7 PPR. Changing PPR does not change the motor's actual speed. Each RPM line
shows the configured PPR. These settings reset on reboot; update the source
constants once confirmed.

At 380:1 gearing, 3 PPR produces 4,560 x4 counts per output revolution; 7 PPR
produces 10,640. To distinguish them, mark the output shaft, zero its count,
and record the change over a known number of complete shaft revolutions.
Divide by the number of revolutions and compare with these values. Do not
force a gearbox that cannot be back-driven; use powered rotation and record
counts at the marked positions instead. Comparing against the nominal 40 RPM
rating alone does not establish the correct PPR.

If a motor does not turn, check VM, common ground, sleep state, and duty. If it
turns but counts stay fixed, check encoder power, signal levels, pull-ups, and
C1/C2 wiring. Counts changing at rest suggest noise or floating inputs.

References: [DRV8833 datasheet](https://www.ti.com/lit/ds/symlink/drv8833.pdf),
[ESP32 GPIO documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/gpio.html).
