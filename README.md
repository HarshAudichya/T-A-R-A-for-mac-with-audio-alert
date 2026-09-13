# TARA — Task-aware Astronaut Recognition Assistant

**TARA (Task-aware Astronaut Recognition Assistant)** is an on-board AI system designed for human activity recognition (HAR) and step-by-step experiment assistance during space missions (such as ISRO Biological and Astronautical Science experiments).

TARA automatically monitors astronaut activities in real time using computer vision (MediaPipe Pose Landmarks and YOLOv8 object detection), tracks adherence to experiment checklists, provides real-time voice and visual guidance, and generates audit reports.

## Features

- **Real-Time Human Activity Recognition (HAR)**: Automatic pose and gesture recognition using MediaPipe.
- **Multimodal Object Tracking**: YOLOv8 and color-aware detection for required experiment payload items (boxes, sample vials, tools).
- **Task-Aware Guidance Engine**: Step-by-step sequence tracking against mission experiment protocols with voice prompts.
- **Mission Debrief & Audit Reports**: Generates detailed HTML & PDF-ready audit reports with synthesized audio debriefing.
- **Spacecraft Glassmorphic HUD**: Streamlit-powered futuristic mission dashboard with real-time telemetry.

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
streamlit run app.py
```

Allow camera access in the browser when prompted.
# T-A-R-A-for-mac-with-audio-alert
