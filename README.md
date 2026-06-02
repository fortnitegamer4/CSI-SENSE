**CSI-Sense**

Detecting Presence Behind Obstructions for Search & Rescue Using WiFi CSI

CSI-Sense is a low-cost WiFi sensing system that uses Channel State Information (CSI) to detect human presence behind obstructions such as doors, walls, and cluttered environments. The project explores how commodity WiFi hardware can assist search-and-rescue (SAR) operations by identifying motion and occupancy without requiring direct line-of-sight.

**Overview**

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


**Running the Project**

To use CSI-Sense, first set up the Python environment and install the required dependencies. The project was developed using a virtual environment to keep package versions isolated.

Next, configure the Raspberry Pi with Nexmon CSI and connect it to the WiFi sensing setup. The Raspberry Pi acts as the receiver while a 2.4 GHz router continuously transmits packets through the monitored region.

Once the hardware is configured, collect CSI data on the Raspberry Pi. Each recording is saved as a PCAP file containing packet-level Channel State Information measurements. Recordings can be collected under vacant conditions or while a person is present behind an obstruction.

After data collection, transfer the PCAP files from the Raspberry Pi to the laptop. The provided conversion script converts each PCAP file into a PKL file containing a CSI matrix and timestamp array. These PKL files serve as the input for the machine learning pipeline.

The training script processes the PKL files, extracts frequency-domain features using a Short-Time Fourier Transform (STFT), and trains a Random Forest classifier to distinguish between vacant and occupied environments. The trained model is saved for future inference.

To evaluate the system, run the classification pipeline on a set of PKL files. The classifier generates window-level occupied probabilities and aggregates them into a final file-level prediction of either VACANT or OCCUPIED.

The project also includes a visualization tool that overlays a red intensity map onto a synchronized phone recording. The intensity is computed from the model's occupied probability and the amount of motion-related energy present in the CSI signal. Higher intensity values indicate stronger evidence of human presence or movement behind the obstruction.

Finally, the evaluation scripts can be used to generate accuracy metrics and comparison graphs showing performance across baseline, simple, moderate, and complex obstruction conditions.

This workflow follows the complete CSI-Sense pipeline:

CSI Capture → PCAP Conversion → Feature Extraction → Model Classification → Visualization → Evaluation.
