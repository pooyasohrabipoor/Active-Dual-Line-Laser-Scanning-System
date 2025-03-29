# Active-Dual-Line-Laser-Scanning-System
This repository is part of my master’s research at the University of Arkansas, focusing on developing a low-cost, Arduino-based active sensing system for 3D vision applications in robotics.

The primary goal is to control a line laser using a servo-actuated mirror, enabling dynamic projection of the laser across a scene. By observing the displacement of the laser line relative to a calibrated baseline, the system can infer depth information — a key step in reconstructing 3D geometry of objects.
Key Concepts:
The mirror angle is controlled via PWM signals from an Arduino, which determines the laser projection angle.

The baseline laser position is mapped based on known mirror angles.

When an object is introduced into the field, the laser line shifts from its baseline.

This shift is used to compute depth, enabling 3D surface profiling.

The given diagram demonstrated the process:

![System Diagram](d.png)

