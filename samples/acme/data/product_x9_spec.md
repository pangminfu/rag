# X-9 Inspection Drone — Product Specification

The **X-9 Inspection Drone** is a compact quadcopter designed for indoor and confined-space industrial inspection — for example, inside boilers, pipelines, storage tanks, and ship hulls. It is built around a protective carbon-fiber cage that allows the drone to bump into surfaces without damage.

## Physical Specifications

| Attribute | Value |
|---|---|
| Model number | X-9 |
| Diameter (with cage) | 360 mm |
| Height | 110 mm |
| Weight (with battery) | 780 g |
| Maximum payload | 120 g |
| Battery | 4S 1500 mAh LiPo (swappable) |
| Flight time per battery | Up to 14 minutes |
| Maximum airspeed | 8 m/s |
| Operating temperature | -10 °C to +45 °C |
| Ingress protection | IP43 |

## Sensors and Cameras

- 4K front-facing inspection camera with 30× digital zoom and adjustable LED ring light (5,000 lumens total).
- 360° lateral collision sensors (8× ToF).
- Downward optical flow + dual ultrasonic for stable hover without GPS.
- Onboard 5.8 GHz video downlink with sub-100 ms latency.

GPS is **not** required for operation, making the X-9 well-suited for fully enclosed spaces where satellite signals are unavailable.

## Operation

The X-9 ships with the AcmePilot ground controller, a tablet-based interface that records flight telemetry, captures still images, and streams live video. Inspection missions can be planned in advance using the AcmePilot mission editor, including waypoints expressed relative to a visually-detected QR-coded fiducial.

A trained operator certificate is required to operate the X-9 in regulated jurisdictions. Acme Robotics provides a one-day operator training course.

## Software

- Runs AcmePilot Firmware 2.x.
- USB-C data export of inspection footage and per-frame metadata (battery, IMU, lighting).
- Optional integration with the Acme Inspect cloud service for AI-based defect detection (corrosion, cracks, weld faults).

## Charging and Battery Care

Each battery includes an integrated balance circuit and reports its cycle count. Batteries should be stored at 40–60% charge if unused for more than 7 days. The X-9 supports hot-swappable batteries, so a typical inspection kit ships with 4 batteries and a 4-bay charger to allow continuous operation.

## Warranty

The X-9 is covered by a **1-year limited warranty** from the date of purchase. The warranty covers manufacturing defects in the airframe, cage, motors, and electronics under normal use. The warranty does **not** cover:

- Crash damage caused by operator error or flying outside published environmental limits.
- Battery degradation after more than 200 charge cycles.
- Water ingress beyond the IP43 rating.
- Damage caused by aftermarket payloads or modifications.

Warranty claims may be initiated through `support@acme-robotics.internal`. For mission-critical fleets, customers can purchase **AcmeCare Air**, which adds a second year of coverage and a 48-hour replacement guarantee.

## Spare Parts

Common spares — propellers, cage segments, and batteries — are stocked at Acme regional depots and ship within 2 business days.
