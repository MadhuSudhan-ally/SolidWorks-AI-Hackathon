# SolidWorks-AI-Hackathon
YOLOv8-based object detection system to identify and count mechanical parts (bolt, nut, washer, locating pin). Achieved 100% exact-match accuracy in SOLIDWORKS AI Hackathon.



Winner solution for the **SOLIDWORKS AI Hackathon (IIT Madras, Kaggle)**.

This project uses **YOLOv8 object detection** to identify and count mechanical components from synthetic CAD images, achieving **100% exact-match accuracy** on the leaderboard.

---

##  Problem Statement
Given synthetic images containing mechanical parts, detect and count:
- Bolt
- Locating Pin
- Nut
- Washer

Each image may contain multiple instances of each object.

---

##  Solution Overview
- Converted bounding-box annotations to YOLO format
- Trained YOLOv8 
- Performed object detection on test images
- Counted detections per class
- Generated competition-ready submission file

---

##  Tech Stack
- Python
- PyTorch
- YOLOv8 (Ultralytics)
- Computer Vision
- Google Colab (T4 GPU)
- Kaggle

---
