CSI-Sense

Detecting Presence Behind Obstructions for Search & Rescue Using WiFi CSI

CSI-Sense is a low-cost WiFi sensing system that uses Channel State Information (CSI) to detect human presence behind obstructions such as doors, walls, and cluttered environments. The project explores how commodity WiFi hardware can assist search-and-rescue (SAR) operations by identifying motion and occupancy without requiring direct line-of-sight.

Overview

Traditional search-and-rescue operations often require responders to physically enter potentially hazardous environments before determining whether victims are present. CSI-Sense investigates whether WiFi signals can be used as a non-invasive sensing modality for presence detection.

The system consists of:

A 2.4 GHz WiFi router acting as the transmitter
A Raspberry Pi 4B running Nexmon CSI firmware as the receiver
A machine learning pipeline for CSI preprocessing and classification
A visualization module that converts model outputs into a real-time intensity overlay
System Pipeline
WiFi Router
     ↓
Signal passes through obstruction
     ↓
Raspberry Pi captures CSI
     ↓
PCAP files recorded
     ↓
PCAP → PKL conversion
     ↓
Feature extraction (STFT)
     ↓
Random Forest classification
     ↓
Vacant / Occupied prediction
     ↓
Visualization & intensity overlay
Hardware
Receiver
Raspberry Pi 4 Model B
Nexmon CSI firmware
Transmitter
2.4 GHz WiFi Router
Processing
Laptop/Desktop running Python
