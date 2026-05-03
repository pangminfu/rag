# R-200 Cleaning Robot — Product Specification

The **R-200 Cleaning Robot** is Acme Robotics' flagship autonomous floor-cleaning platform, designed for medium-sized commercial environments such as offices, retail stores, and small warehouses (up to 5,000 m²). It combines LIDAR-based navigation with a dual-mode dry/wet cleaning head.

## Physical Specifications

| Attribute | Value |
|---|---|
| Model number | R-200 |
| Dimensions (L × W × H) | 620 × 540 × 280 mm |
| Weight (with empty tank) | 28 kg |
| Cleaning width | 480 mm |
| Water tank capacity | 4.5 L |
| Debris bin capacity | 3.0 L |
| Battery | 36 V, 12 Ah lithium-ion |
| Run time per charge | Up to 4 hours |
| Charge time | 3 hours (standard dock), 90 minutes (fast-charge dock) |
| Maximum slope | 8° |
| Operating noise | 62 dB(A) |
| Ingress protection | IP44 |

## Cleaning Modes

- **Dry mode:** brushroll + HEPA vacuum, suitable for carpets and hard floors.
- **Wet mode:** controlled water dispense + microfiber pad, suitable for sealed hard floors only.
- **Hybrid mode:** sequential vacuum-then-mop pass.

The R-200 will automatically pause wet mode if it detects a carpeted area via its onboard floor classifier.

## Navigation and Sensors

- 360° single-plane LIDAR (12 m range)
- Front and side time-of-flight bumpers
- Downward-facing cliff sensors (4×)
- 1080p front camera for object recognition (people, chairs, cables)
- IMU + wheel encoders for dead reckoning
- Wi-Fi 6 (2.4/5 GHz) and Bluetooth 5.2 for fleet management

The R-200 builds and stores up to 10 floor-plan maps onboard. Maps can be edited from the Acme Fleet Console to add no-go zones and scheduling rules.

## Software

- Operates on AcmeOS 4.x.
- OTA firmware updates pushed via the Fleet Console.
- REST API + MQTT bridge for integration with building-management systems.

## Power and Charging

The R-200 returns to its dock automatically when the battery falls below 15% or at the end of a cleaning cycle. The dock requires a standard 100–240 V AC outlet.

## Warranty

The R-200 is covered by a **2-year limited warranty** from the date of purchase. The warranty covers manufacturing defects and component failures under normal use, including the battery, motors, sensors, and main control board. The warranty does **not** cover:

- Damage caused by liquids other than clean water in the tank.
- Damage from drops, collisions, or unauthorized modifications.
- Wear-and-tear consumables: brushrolls, microfiber pads, and HEPA filters.

To make a warranty claim, contact `support@acme-robotics.internal` with the unit's serial number and a description of the issue. Replacement parts are typically shipped within 5 business days.

## Service Plan (Optional)

Customers may purchase **AcmeCare Plus**, which extends the warranty to a total of 4 years and includes one free annual maintenance visit by a certified Acme technician.
