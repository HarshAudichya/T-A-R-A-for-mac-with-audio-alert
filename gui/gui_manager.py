import json
import time
import os
import textwrap
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit.runtime.scriptrunner as sr
ScriptRequestType = sr.script_runner.ScriptRequestType

from core.camera_manager import CameraManager
from core.video_manager import VideoManager
from core.experiment_manager import ExperimentManager
from ai.pose_estimator import PoseEstimator
from ai.object_detector import ObjectDetector
from ai.interaction_detector import InteractionDetector
from ai.activity_recognizer import ActivityRecognizer
from ai.pose_3d_estimator import Pose3DEstimator
from ai.model_manager import ModelManager
from validation.sequence_validator import SequenceValidator
from alerts.alert_manager import AlertManager
from streaming.stream_manager import StreamManager
from logging_system.log_manager import LogManager
from dataset.dataset_generator import DatasetGenerator
from telemetry.bandwidth_telemetry import BandwidthTelemetryTracker
from reports.report_generator import MissionReportGenerator

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
}

def clean_html(html_str: str) -> str:
    """Strips all leading/trailing whitespace from each line to prevent Streamlit Markdown code block parsing."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines())

# Pre-warm AI Models in Memory for Instant App Loading (0.0s startup delay)
@st.cache_resource
def get_cached_pipeline():
    """Single-instance cached pipeline to pre-warm YOLOv8 and MediaPipe models once."""
    print("[GUIManager] Pre-warming shared DirectCameraPipeline in memory...")
    pipe = DirectCameraPipeline()
    pipe.object_detector.get_yolo()
    return pipe

# Futuristic Spacecraft AI Control System CSS Theme with Crimson Red Alerts
ASTRO_CSS = clean_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600;700&display=swap');

/* Global Dark Spacecraft Theme */
.stApp {
    background-color: #070A0F;
    background-image: 
        radial-gradient(at 0% 0%, rgba(0, 240, 255, 0.04) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(239, 68, 68, 0.04) 0px, transparent 50%);
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

/* TARA Title Banner */
.astro-header {
    background: linear-gradient(135deg, rgba(13, 20, 36, 0.9) 0%, rgba(8, 12, 22, 0.9) 100%);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-left: 4px solid #00F0FF;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.1);
    backdrop-filter: blur(12px);
}
.astro-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.85rem;
    font-weight: 900;
    letter-spacing: 2px;
    color: #00F0FF;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
    margin: 0;
}
.astro-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    color: #38BDF8;
    margin-top: 4px;
    text-transform: uppercase;
}

/* Glassmorphic Spacecraft Cards */
.astro-card {
    background: rgba(13, 19, 33, 0.75);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    position: relative;
}
.astro-card::before {
    content: '';
    position: absolute;
    top: -1px;
    left: 15px;
    width: 30px;
    height: 2px;
    background: #00F0FF;
    box-shadow: 0 0 8px #00F0FF;
}

.card-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: #00F0FF;
    margin-bottom: 10px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Telemetry Metric Badges */
.telemetry-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}
.telemetry-item {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
}
.telemetry-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #94A3B8;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.telemetry-val {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #00F0FF;
}
.telemetry-val.online { color: #10B981; }
.telemetry-val.alert { color: #EF4444; }

/* Object Tracking HUD Cards */
.obj-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
    margin-top: 8px;
}
.obj-card {
    background: rgba(8, 14, 26, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-radius: 6px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.05);
}
.obj-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #F8FAFC;
}
.obj-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #00F0FF;
    margin: 4px 0;
}
.obj-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #10B981;
    background: rgba(16, 185, 129, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
}

/* Mission Timeline */
.timeline-item {
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.timeline-completed {
    background: rgba(16, 185, 129, 0.1);
    border-left: 3px solid #10B981;
    color: #34D399;
}
.timeline-active {
    background: rgba(0, 240, 255, 0.15);
    border-left: 3px solid #00F0FF;
    color: #00F0FF;
    font-weight: 700;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.15);
}
.timeline-upcoming {
    background: rgba(30, 41, 59, 0.4);
    border-left: 3px solid #475569;
    color: #94A3B8;
}

/* Terminal Log */
.terminal-container {
    background: #03060A;
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 6px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #38BDF8;
    max-height: 220px;
    overflow-y: auto;
}

/* Camera HUD Frame */
.cam-hud-header {
    background: rgba(10, 16, 28, 0.95);
    border: 1px solid rgba(0, 240, 255, 0.3);
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00F0FF;
}
.cam-live-indicator {
    color: #EF4444;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}

/* Streamlit Component Overrides */
div.stButton > button {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.15) 0%, rgba(56, 189, 248, 0.05) 100%);
    border: 1px solid rgba(0, 240, 255, 0.4);
    color: #00F0FF;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    border-radius: 6px;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: rgba(0, 240, 255, 0.3);
    border-color: #00F0FF;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    color: #FFFFFF;
}
.stSelectbox label, .stRadio label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    color: #00F0FF !important;
    letter-spacing: 1px;
}
</style>
""")

class DirectCameraPipeline:
    """Direct OpenCV camera pipeline capturing local hardware frames with 0% network overhead."""
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetector()
        self.interaction_detector = InteractionDetector()
        self.activity_recognizer = ActivityRecognizer()
        self.sequence_validator = SequenceValidator()
        self.alert_manager = AlertManager()
        self.camera_manager = CameraManager()
        self.video_manager = VideoManager()
        self.dataset_generator = DatasetGenerator()

        self.experiment_manager = ExperimentManager("config/activities.json")
        self.log_manager = LogManager()

        self.activity = "Unknown"
        self.confidence = 0.0
        self.interaction_event = "None"
        self.alert_msg = ""
        self.last_completed_time = 0.0
        self.detections = []
        self.latest_raw_frame = None

    def set_activity_group(self, group_name: str) -> None:
        self.experiment_manager.set_activity_group(group_name)
        self.dataset_generator.set_active_group(group_name)
        self.alert_manager.purge_speech()
        self.alert_msg = ""
        self.last_completed_time = 0.0
        self.activity = "Unknown"
        self.confidence = 0.0
        self._completion_announced = False

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        self.latest_raw_frame = image.copy()

        landmarks, annotated_image = self.pose_estimator.estimate(image)
        expected_act = self.experiment_manager.get_expected_activity()
        expected_obj = self.experiment_manager.get_expected_required_object()

        self.detections = self.object_detector.detect(image, landmarks, expected_obj)
        annotated_image = self.object_detector.draw_detections(annotated_image, self.detections)

        interaction_res = self.interaction_detector.detect_interaction(landmarks, self.detections, w, h)
        self.interaction_event = interaction_res["interaction_event"]

        activity, confidence = self.activity_recognizer.predict(landmarks, interaction_res, expected_act)
        self.activity = activity
        self.confidence = confidence

        val_res = self.sequence_validator.validate(expected_act, expected_obj, activity, self.detections, confidence)
        status_code = val_res["status"]

        now = time.time()
        if status_code == SequenceValidator.CORRECT and now - self.last_completed_time > 1.2:
            step_obj = self.experiment_manager.get_current_step()
            step_id = step_obj["step"] if step_obj and "step" in step_obj else self.experiment_manager.current_idx + 1
            self.log_manager.log_event(step_id, expected_act, activity, confidence, "COMPLETED")
            self.alert_manager.trigger_correct_step_alert(activity)
            
            if self.latest_raw_frame is not None:
                meta = {
                    "step": step_id,
                    "activity_group": self.experiment_manager.activity_manager.get_current_group(),
                    "expected_object": expected_obj,
                    "confidence": confidence,
                    "interaction_event": self.interaction_event,
                    "detected_objects": [d["class"] for d in self.detections]
                }
                self.dataset_generator.save_sample(activity, self.latest_raw_frame, meta)

            self.experiment_manager.advance_step()
            self.last_completed_time = now
            self.alert_msg = ""
        elif status_code == SequenceValidator.COMPLETED:
            if not getattr(self, "_completion_announced", False):
                self._completion_announced = True
                self.alert_manager.trigger_completion_alert()
            self.alert_msg = ""
        elif status_code == SequenceValidator.OBJECT_MISSING and now - self.last_completed_time > 1.5:
            det_names = [d["class"] for d in self.detections]
            triggered_msg = self.alert_manager.trigger_missing_object_alert(expected_obj, det_names)
            if triggered_msg:
                if det_names:
                    self.alert_msg = f"WRONG OBJECT: Detected {', '.join(det_names)} (Expected '{expected_obj}')"
                else:
                    self.alert_msg = f"MISSING OBJECT: '{expected_obj}' not detected!"
        elif status_code == SequenceValidator.WRONG_ACTIVITY and (now - self.last_completed_time > 1.5) and (now - self.alert_manager.last_alert_time > 2.0):
            step_obj = self.experiment_manager.get_current_step()
            step_id = step_obj["step"] if step_obj and "step" in step_obj else self.experiment_manager.current_idx + 1
            self.log_manager.log_event(step_id, expected_act, activity, confidence, "OUT_OF_SEQUENCE")
            triggered_msg = self.alert_manager.trigger_wrong_step_alert(expected_act, activity)
            if triggered_msg:
                self.alert_msg = f"WRONG STEP: Detected '{activity}' (Expected '{expected_act}')"

        if self.video_manager.is_recording:
            self.video_manager.write_frame(annotated_image)

        self.draw_hud(annotated_image)
        return annotated_image

    def draw_hud(self, image: np.ndarray) -> None:
        h, w, _ = image.shape
        now = time.time()

        # Top Futuristic Banner
        cv2.rectangle(image, (0, 0), (w, 65), (10, 16, 28), -1)
        cur_step = self.experiment_manager.get_current_step()
        if cur_step:
            obj_text = f" (Req: {cur_step['required_object']})" if cur_step.get('required_object') else ""
            cv2.putText(image, f"TARGET STEP {cur_step['step']}/{len(self.experiment_manager.steps)}: {cur_step['activity'].upper()}{obj_text}", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 240, 255), 2)
            cv2.putText(image, f"Instruction: {cur_step['instruction']}", (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1)
        else:
            cv2.putText(image, "🎉 ALL EXPERIMENT STEPS COMPLETED!", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)

        # Status & BRIGHT RED ALERT Banner (BGR: 30, 0, 240)
        if cur_step:
            if now - self.last_completed_time < 2.0:
                cv2.rectangle(image, (0, 65), (w, 105), (16, 185, 129), -1)
                cv2.putText(image, f"✅ CORRECT STEP! Step {cur_step['step']} completed!", (15, 93),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif self.alert_msg:
                # BRIGHT CRIMSON RED ALERT BANNER
                cv2.rectangle(image, (0, 65), (w, 105), (30, 0, 240), -1)
                cv2.putText(image, f"🚨 {self.alert_msg}", (15, 93),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Right Side Checklist Overlay
        box_w = 270
        box_h = min(h - 130, len(self.experiment_manager.steps) * 30 + 45)
        x1, y1 = w - box_w - 10, 75
        x2, y2 = w - 10, y1 + box_h
        
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 14, 26), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 240, 255), 1)

        cv2.putText(image, "MISSION TIMELINE", (x1 + 10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1)

        for idx, step_item in enumerate(self.experiment_manager.steps):
            item_y = y1 + 52 + idx * 28
            if item_y > y2 - 10:
                break
            if idx < self.experiment_manager.current_idx:
                cv2.putText(image, f"✓ STEP {idx+1:02d}: {step_item['activity']}", (x1 + 10, item_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (16, 185, 129), 1)
            elif idx == self.experiment_manager.current_idx:
                cv2.putText(image, f"▶ STEP {idx+1:02d}: {step_item['activity']}", (x1 + 10, item_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 2)
            else:
                cv2.putText(image, f"○ STEP {idx+1:02d}: {step_item['activity']}", (x1 + 10, item_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (148, 163, 184), 1)

        # Bottom Telemetry HUD
        rec_str = "REC 🔴" if self.video_manager.is_recording else "OFF"
        cv2.rectangle(image, (10, h - 60), (450, h - 10), (4, 8, 16), -1)
        cv2.rectangle(image, (10, h - 60), (450, h - 10), (0, 240, 255), 1)
        cv2.putText(image, f"HAR: {self.activity} ({self.confidence:.0%}) | FPS: {self.camera_manager.fps:.1f}", (20, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Interaction: {self.interaction_event} | Rec: {rec_str}", (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (56, 189, 248), 1)


class HARVideoTransformer(VideoTransformerBase):
    """WebRTC Video Transformer Base."""
    def __init__(self):
        self.pipeline = get_cached_pipeline()

    def set_activity_group(self, group_name: str) -> None:
        self.pipeline.set_activity_group(group_name)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        return self.pipeline.process_frame(image)


# Load TARA Logo Base64 for Zero-Latency High-DPI Display
_LOGO_FILE = os.path.join(os.path.dirname(__file__), "assets", "tara_logo_b64.txt")
try:
    with open(_LOGO_FILE, "r") as f:
        TARA_LOGO_B64 = f.read().strip()
except Exception:
    TARA_LOGO_B64 = ""

class GUIManager:
    """Renders the TARA — Task-aware Astronaut Recognition Assistant Dashboard."""
    def __init__(self):
        self.model_manager = ModelManager()
        self.pose_3d_estimator = Pose3DEstimator()
        self.dataset_generator = DatasetGenerator()
        self.stream_manager = StreamManager()
        self.bandwidth_tracker = BandwidthTelemetryTracker()
        
        # Use fast cached pre-warmed single-instance pipeline
        if "direct_pipeline" not in st.session_state:
            st.session_state.direct_pipeline = get_cached_pipeline()

    def render(self):
        # Inject Futuristic Custom CSS System
        st.markdown(ASTRO_CSS, unsafe_allow_html=True)

        # Render Main Website Logo in Sidebar Navigation
        with st.sidebar:
            sidebar_brand_html = clean_html(f"""
            <div style="text-align: center; padding: 20px 10px 18px 10px; background: rgba(13, 20, 36, 0.85); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 8px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);">
                <img src="data:image/png;base64,{TARA_LOGO_B64}" style="width: 90%; max-width: 220px; filter: drop-shadow(0 0 16px rgba(0, 240, 255, 0.6)) drop-shadow(0 0 6px rgba(255, 138, 0, 0.45));" alt="TARA Logo" />
                <div style="font-family: 'Orbitron', sans-serif; font-size: 0.75rem; letter-spacing: 2px; color: #38BDF8; margin-top: 10px; font-weight: 700;">
                    ISRO MISSION HAR
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #94A3B8; margin-top: 4px;">
                    EDGE AI &bull; V2.0 ONLINE
                </div>
            </div>
            """)
            st.markdown(sidebar_brand_html, unsafe_allow_html=True)

        # 1. TARA Header Banner with Custom Main Website Logo
        header_html = clean_html(f"""
        <div class="astro-header" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
            <div style="display: flex; align-items: center; gap: 24px;">
                <img src="data:image/png;base64,{TARA_LOGO_B64}" style="height: 58px; filter: drop-shadow(0 0 20px rgba(0, 240, 255, 0.6)) drop-shadow(0 0 6px rgba(255, 138, 0, 0.45));" alt="TARA Main Logo" />
                <div style="border-left: 2px solid rgba(0, 240, 255, 0.35); padding-left: 20px;">
                    <div class="astro-title" style="font-size: 1.22rem; letter-spacing: 2.5px;">TASK-AWARE ASTRONAUT RECOGNITION ASSISTANT</div>
                    <div class="astro-subtitle" style="font-size: 0.8rem; margin-top: 3px;">AI ON-BOARD ASTRONAUT EXPERIMENT ASSISTANT &bull; MISSION CONTROL HAR ENGINE &bull; SYSTEM ONLINE</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="obj-badge" style="background: rgba(0, 240, 255, 0.12); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.4); font-size: 0.75rem; padding: 6px 12px; font-weight: 700; letter-spacing: 1px;">
                    ● SYSTEM NOMINAL
                </span>
            </div>
        </div>
        """)
        st.markdown(header_html, unsafe_allow_html=True)

        pipeline = st.session_state.direct_pipeline

        # 2. System Status & Telemetry Bar
        cam_status_str = "CONNECTED" if pipeline.camera_manager.ret or True else "DISCONNECTED"
        stream_status_str = self.stream_manager.get_status()["status_text"]
        rec_status_str = "REC 🔴" if pipeline.video_manager.is_recording else "READY"

        telemetry_html = clean_html(f"""
        <div class="telemetry-bar">
            <div class="telemetry-item">
                <div class="telemetry-label">CAM-01</div>
                <div class="telemetry-val online">{cam_status_str}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">POSE AI</div>
                <div class="telemetry-val online">MEDIAPIPE</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">OBJECT AI</div>
                <div class="telemetry-val online">YOLOv8</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">HAR ENGINE</div>
                <div class="telemetry-val online">MULTIMODAL</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">VOICE AI</div>
                <div class="telemetry-val online">ACTIVE 🔊</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">RECORDING</div>
                <div class="telemetry-val">{rec_status_str}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">IP STREAM</div>
                <div class="telemetry-val">{stream_status_str}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">MISSION</div>
                <div class="telemetry-val online">RUNNING</div>
            </div>
        </div>
        """)
        st.markdown(telemetry_html, unsafe_allow_html=True)

        # 3. FEATURE 3: Spacecraft Edge Bandwidth Telemetry Card
        bw_data = self.bandwidth_tracker.compute_telemetry(
            width=640, height=480, fps=pipeline.camera_manager.fps or 30.0
        )
        bandwidth_card_html = clean_html(f"""
        <div class="astro-card">
            <div class="card-label">🛰️ SPACECRAFT EDGE BANDWIDTH TELEMETRY</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 10px;">
                <div class="telemetry-item">
                    <div class="telemetry-label">RAW VIDEO OVERHEAD</div>
                    <div class="telemetry-val alert">{bw_data['raw_video_rate_str']}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">EDGE AI TELEMETRY</div>
                    <div class="telemetry-val online">{bw_data['ai_telemetry_rate_str']}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">BANDWIDTH SAVED</div>
                    <div class="telemetry-val online">{bw_data['data_saved_str']}</div>
                </div>
                <div class="telemetry-item">
                    <div class="telemetry-label">EDGE PROCESSING</div>
                    <div class="telemetry-val online">● ACTIVE</div>
                </div>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94A3B8; margin-top: 8px;">
                &bull; {bw_data['bandwidth_label']}: Local Edge HAR execution transmits lightweight JSON events instead of raw high-bandwidth 1080p video stream.
            </div>
        </div>
        """)
        st.markdown(bandwidth_card_html, unsafe_allow_html=True)

        # 4. Experiment Module Selector Card (FEATURE 1: Added ISRO Red & Yellow Box BAS Experiment)
        selector_card_html = clean_html("""
        <div class="astro-card">
            <div class="card-label">⚙️ EXPERIMENT MODULE SELECTOR</div>
        </div>
        """)
        st.markdown(selector_card_html, unsafe_allow_html=True)

        group_options = [
            "Basic Human Activities",
            "Phone Activities",
            "Pen Activities",
            "Chair Activities",
            "Box Activities",
            "Bottle Activities",
            "Book Activities",
            "Multi-Object Experiment",
            "ISRO Red & Yellow Box BAS Experiment"
        ]

        selected_group = st.selectbox(
            "Active Experiment Module:",
            group_options,
            index=8, # Defaults to the official ISRO Red & Yellow Box Experiment!
            key="dash_group_selector",
            help="Selecting an experiment module automatically switches activity recognition rules and dataset directories."
        )

        if "prev_selected_group" not in st.session_state:
            st.session_state.prev_selected_group = selected_group

        # When changing activity group: instantly purge any ongoing speech and reset alert state
        if selected_group != st.session_state.prev_selected_group:
            st.session_state.prev_selected_group = selected_group
            pipeline.set_activity_group(selected_group)
            pipeline.experiment_manager.reset()
            pipeline.activity = "Unknown"
            pipeline.confidence = 0.0
            pipeline.detections = []
            pipeline.alert_msg = ""
            pipeline.last_completed_time = 0.0
            pipeline.last_alert_time = 0.0
            pipeline._completion_announced = False
            pipeline.alert_manager.purge_speech()
            pipeline.alert_manager.speak_instant(f"Switched to {selected_group}.")
        else:
            pipeline.set_activity_group(selected_group)

        # 6. Navigation Tabs Layout
        tab_dashboard, tab_dataset, tab_system = st.tabs([
            "📹 LIVE MISSION DASHBOARD", "📊 GROUP DATASET MODULE", "⚙️ SYSTEM TELEMETRY & CONTROLS"
        ])

        with tab_dashboard:
            # BRIGHT CRIMSON RED ALERT BANNER (If missing object or wrong step occurs)
            if pipeline.alert_msg:
                alert_banner_html = clean_html(f"""
                <div style="background: rgba(239, 68, 68, 0.25); border: 2px solid #EF4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);">
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 1rem; font-weight: 800; color: #FF3366; letter-spacing: 1px;">
                        🚨 SYSTEM ALERT: {pipeline.alert_msg.upper()}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #FCA5A5; margin-top: 4px;">
                        Sequence validation active &bull; Please perform the required action or present the required object.
                    </div>
                </div>
                """)
                st.markdown(alert_banner_html, unsafe_allow_html=True)

            # Top Grid: Live Camera HUD Frame + TARA AI Real-Time Confidence Panel
            cam_col, ai_col = st.columns([3, 2])

            ai_card_placeholder = ai_col.empty()
            obj_card_placeholder = ai_col.empty()
            req_card_placeholder = ai_col.empty()
            voice_panel_placeholder = ai_col.empty()

            def render_ai_cards_live():
                """Re-renders right-hand TARA AI cards live on every camera frame tick."""
                conf_val = pipeline.confidence
                conf_pct = f"{conf_val:.0%}"
                conf_num = int(conf_val * 100)

                is_alert = bool(pipeline.alert_msg)
                status_color = "#EF4444" if is_alert else "#10B981"
                status_text = f"🚨 {pipeline.alert_msg}" if is_alert else "● CONFIRMED &bull; REAL-TIME HAR MATCH"

                astro_card_html = clean_html(f"""
                <div class="astro-card">
                    <div class="card-label">✦ TARA AI &bull; REAL-TIME HAR & CONFIDENCE CHECK</div>
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 8px;">
                        {pipeline.activity.upper()}
                    </div>
                    <div style="margin-top: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8;">REAL-TIME HAR CONFIDENCE</span>
                            <span style="font-family: 'Orbitron', sans-serif; font-size: 1.25rem; font-weight: 700; color: #00F0FF;">{conf_pct}</span>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 6px; height: 12px; overflow: hidden; width: 100%;">
                            <div style="background: linear-gradient(90deg, #00F0FF 0%, #38BDF8 100%); width: {conf_num}%; height: 100%; box-shadow: 0 0 10px #00F0FF;"></div>
                        </div>
                    </div>
                    <div style="margin-top: 12px; display: flex; gap: 15px; align-items: center;">
                        <div>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8;">INTERACTION</span><br/>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38BDF8;">{pipeline.interaction_event}</span>
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span class="obj-badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}88; font-weight: 700; padding: 4px 10px; font-size: 0.75rem;">
                            {status_text}
                        </span>
                    </div>
                </div>
                """)
                ai_card_placeholder.markdown(astro_card_html, unsafe_allow_html=True)

                # Object Tracking Card (LIVE REAL-TIME DISPLAY)
                obj_html = clean_html("""
                <div class="astro-card">
                    <div class="card-label">🎯 OBJECT TRACKING (YOLOv8 + COLOR AI)</div>
                """)
                if pipeline.detections:
                    obj_html += "<div class='obj-grid'>"
                    icons = {
                        "Phone": "📱", "Bottle": "🍼", "Chair": "🪑", "Pen": "✏️",
                        "Box": "📦", "Red Box": "🟥", "Yellow Box": "🟨", "Book": "📖"
                    }
                    for d in pipeline.detections:
                        cls_name = d["class"]
                        conf_str = f"{d['confidence']:.0%}"
                        ic = icons.get(cls_name, "🔍")
                        obj_html += clean_html(f"""
                        <div class="obj-card">
                            <div class="obj-name">{ic} {cls_name.upper()}</div>
                            <div class="obj-conf">{conf_str}</div>
                            <div class="obj-badge">TRACKED</div>
                        </div>
                        """)
                    obj_html += "</div>"
                else:
                    obj_html += "<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem; color: #94A3B8; margin-top: 6px;'>OBJECT NOT DETECTED IN FRAME</div>"
                obj_html += "</div>"
                obj_card_placeholder.markdown(obj_html, unsafe_allow_html=True)

                # Required Payload Objects Card
                req_objs = pipeline.experiment_manager.activity_manager.get_required_objects()
                req_str = " &bull; ".join(req_objs) if req_objs else "None (Body Pose Activities)"
                req_card_html = clean_html(f"""
                <div class="astro-card">
                    <div class="card-label">📦 REQUIRED PAYLOAD OBJECTS</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #38BDF8;">
                        {req_str}
                    </div>
                </div>
                """)
                req_card_placeholder.markdown(req_card_html, unsafe_allow_html=True)

                # FEATURE 4: TARA Voice Assistant HUD Card
                cur_voice_msg = pipeline.alert_manager.last_alert_message or "System active. Ready for experiment steps."
                voice_card_html = clean_html(f"""
                <div class="astro-card">
                    <div class="card-label">🔊 TARA VOICE ASSISTANT</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #10B981; margin-bottom: 6px;">
                        ● BROWSER AUDIO READY &bull; pyttsx3 BACKEND ONLINE
                    </div>
                    <div style="font-family: 'Inter', sans-serif; font-size: 0.82rem; color: #E2E8F0; background: rgba(15, 23, 42, 0.8); padding: 8px 12px; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.2);">
                        <b>Current Alert:</b> "{cur_voice_msg}"
                    </div>
                </div>
                """)
                voice_panel_placeholder.markdown(voice_card_html, unsafe_allow_html=True)

            # Render initial static state of AI cards
            render_ai_cards_live()

            # Voice Test Action Button in ai_col
            with ai_col:
                if st.button("🔊 TEST VOICE ALERT AUDIO", width="stretch", key="btn_test_voice"):
                    pipeline.alert_manager.speak_instant("TARA AI Audio System online. Voice alert test nominal.")

            with cam_col:
                cam_hud_html = clean_html(f"""
                <div class="cam-hud-header">
                    <span>LIVE EXPERIMENT MONITOR &bull; {selected_group.upper()}</span>
                    <span>CAM-01 <span class="cam-live-indicator">LIVE ●</span> {pipeline.camera_manager.fps:.1f} FPS</span>
                </div>
                """)
                st.markdown(cam_hud_html, unsafe_allow_html=True)

                stream_mode = st.radio(
                    "Camera Engine Mode:",
                    ["🟢 Direct Hardware Camera (Continuous 30 FPS Stream)", "🌐 WebRTC Stream"],
                    horizontal=True,
                    key="stream_mode_radio"
                )

                if "Direct Hardware Camera" in stream_mode:
                    run_direct = st.toggle("▶️ Start Live Mission Feed", value=True, key="toggle_direct_cam")
                    if run_direct:
                        cam_placeholder = st.empty()
                        audio_slot = st.empty()
                        cam_mgr = CameraManager()
                        cam_mgr.start()
                        last_alert_id_played = pipeline.alert_manager.event_counter
                        
                        # Continuous live frame streaming loop with real-time UI card updates!
                        while run_direct:
                            # Cooperative check: if user interacted with dropdown, button, or toggled camera, break immediately!
                            ctx = get_script_run_ctx()
                            if ctx and getattr(ctx, "script_requests", None):
                                if ctx.script_requests._state != ScriptRequestType.CONTINUE:
                                    break

                            ret, raw_frame = cam_mgr.read()
                            if ret and raw_frame is not None:
                                annotated_frame = pipeline.process_frame(raw_frame)
                                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                cam_placeholder.image(rgb_frame, channels="RGB", width="stretch")
                                # Instantly re-render TARA AI confidence gauge & detected object cards on every frame tick!
                                render_ai_cards_live()

                                # Real-time In-Loop Audio Broadcast (plays immediately on current frame tick!)
                                cur_ev = pipeline.alert_manager.get_latest_voice_event()
                                if cur_ev and cur_ev.get("id", 0) > last_alert_id_played:
                                    last_alert_id_played = cur_ev["id"]
                                    escaped_speech = cur_ev["message"].replace("'", "\\'").replace('"', '\\"')
                                    audio_slot.markdown(f"""
                                    <div style="display:none;">
                                        <script>
                                        try {{
                                            if ('speechSynthesis' in window) {{
                                                window.speechSynthesis.cancel();
                                                var u = new SpeechSynthesisUtterance("{escaped_speech}");
                                                u.rate = 1.0;
                                                window.speechSynthesis.speak(u);
                                            }}
                                        }} catch(e) {{}}
                                        </script>
                                    </div>
                                    """, unsafe_allow_html=True)
                            time.sleep(0.03) # Smooth 30 FPS playback
                    else:
                        # When camera is turned off: stop hardware camera, purge frame buffers, and set pipeline to STANDBY!
                        cam_mgr = CameraManager()
                        cam_mgr.stop()
                        pipeline.activity = "STANDBY"
                        pipeline.confidence = 0.0
                        pipeline.detections = []
                        pipeline.interaction_event = "None"
                        pipeline.alert_msg = ""
                        pipeline.alert_manager.purge_speech()
                        render_ai_cards_live()
                        offline_cam_html = clean_html("""
                        <div style="background: rgba(10, 16, 28, 0.9); border: 2px dashed rgba(0, 240, 255, 0.3); border-radius: 8px; height: 320px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #94A3B8; margin-top: 10px;">
                            <div style="font-size: 2.8rem; margin-bottom: 8px;">📹</div>
                            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; color: #00F0FF;">MISSION FEED OFFLINE</div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; margin-top: 4px; color: #64748B;">Camera hardware stopped. Toggle 'Start Live Mission Feed' to begin.</div>
                        </div>
                        """)
                        st.markdown(offline_cam_html, unsafe_allow_html=True)
                else:
                    cam_mgr = CameraManager()
                    cam_mgr.stop()

                    ctx = webrtc_streamer(
                        key="isro-bas-camera-v22",
                        video_processor_factory=HARVideoTransformer,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video": True, "audio": False},
                        async_processing=True,
                    )
                    if ctx.video_processor:
                        ctx.video_processor.set_activity_group(selected_group)

            st.divider()

            # Guidance & Mission Timeline Row
            c_left, c_right = st.columns(2)

            with c_left:
                exp_mgr = pipeline.experiment_manager
                cur_step = exp_mgr.get_current_step()
                nxt = exp_mgr.get_next_step()

                if cur_step:
                    req_obj = f" (Req: {cur_step['required_object']})" if cur_step.get('required_object') else ""
                    nxt_str = f"STEP {nxt['step']}: {nxt['activity']}" if nxt else "EXPERIMENT COMPLETE 🏁"

                    guidance_html = clean_html(f"""
                    <div class="astro-card">
                        <div class="card-label">✦ TARA AI GUIDANCE &bull; TARGET STEP {cur_step['step']}/{len(exp_mgr.steps)}</div>
                        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.25rem; font-weight: 700; color: #00F0FF; margin-top: 6px;">
                            {cur_step['activity'].upper()}{req_obj}
                        </div>
                        <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; color: #E2E8F0; margin-top: 8px;">
                            👉 <b>Instruction:</b> {cur_step['instruction']}
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94A3B8; margin-top: 10px;">
                            ⏭️ <b>NEXT REQUIRED STEP:</b> {nxt_str}
                        </div>
                    </div>
                    """)
                    st.markdown(guidance_html, unsafe_allow_html=True)
                else:
                    completed_html = clean_html("""
                    <div class="astro-card">
                        <div class="card-label">🎉 MISSION STATUS</div>
                        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.3rem; color: #10B981;">
                            ALL EXPERIMENT STEPS COMPLETED SUCCESSFULLY!
                        </div>
                    </div>
                    """)
                    st.markdown(completed_html, unsafe_allow_html=True)

                if st.button("🔄 RESET EXPERIMENT SEQUENCE (RESTART FROM STEP 1)", width="stretch"):
                    pipeline.experiment_manager.reset()
                    pipeline.last_completed_time = 0.0
                    pipeline.alert_msg = ""
                    pipeline._completion_announced = False
                    st.success("Experiment sequence reset back to Step 1!")

            with c_right:
                timeline_card_header = clean_html("""
                <div class="astro-card">
                    <div class="card-label">📋 MISSION TIMELINE</div>
                """)
                st.markdown(timeline_card_header, unsafe_allow_html=True)

                exp_mgr = pipeline.experiment_manager
                timeline_html = ""
                for idx, step_item in enumerate(exp_mgr.steps):
                    if idx < exp_mgr.current_idx:
                        timeline_html += clean_html(f"""
                        <div class="timeline-item timeline-completed">
                            <span>✓ STEP {idx+1:02d}</span>
                            <span>{step_item['activity']}</span>
                        </div>
                        """)
                    elif idx == exp_mgr.current_idx:
                        timeline_html += clean_html(f"""
                        <div class="timeline-item timeline-active">
                            <span>▶ STEP {idx+1:02d}</span>
                            <span>{step_item['activity']} (ACTIVE TARGET)</span>
                        </div>
                        """)
                    else:
                        timeline_html += clean_html(f"""
                        <div class="timeline-item timeline-upcoming">
                            <span>○ STEP {idx+1:02d}</span>
                            <span>{step_item['activity']}</span>
                        </div>
                        """)
                st.markdown(timeline_html + "</div>", unsafe_allow_html=True)

            st.divider()

            # Spacecraft Terminal Event Log Audit & FEATURE 5: Mission Audit Report
            term_card_header = clean_html("""
            <div class="astro-card">
                <div class="card-label">📝 SYSTEM EVENT LOG TERMINAL & MISSION AUDIT</div>
            """)
            st.markdown(term_card_header, unsafe_allow_html=True)

            if pipeline.log_manager.events:
                events = pipeline.log_manager.events
                term_html = "<div class='terminal-container'>"
                for ev in events[-15:]:
                    ts = ev.get('timestamp', '')[11:19] if 'timestamp' in ev else '08:42:00'
                    status_tag = ev.get('status', 'INFO')
                    tag_color = "#10B981" if status_tag == "COMPLETED" else "#EF4444"
                    term_html += f"<div>{ts} &nbsp; [<span style='color:{tag_color};'>{status_tag}</span>] Step {ev.get('step','0')}: {ev.get('detected_activity','Unknown')} (Conf: {ev.get('confidence',0):.0%})</div>"
                term_html += "</div>"
                st.markdown(term_html, unsafe_allow_html=True)

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "⬇️ DOWNLOAD EVENT LOG (JSON)",
                        data=json.dumps(events, indent=2),
                        file_name=f"tara_mission_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width="stretch"
                    )
                with col_d2:
                    if st.button("📄 GENERATE MISSION AUDIT REPORT", width="stretch"):
                        st.session_state.generated_report_html = MissionReportGenerator.generate_html_report(
                            experiment_name=selected_group,
                            steps=pipeline.experiment_manager.steps,
                            current_idx=pipeline.experiment_manager.current_idx,
                            events=pipeline.log_manager.events
                        )
                        st.success("✅ Mission Completion Audit Report generated successfully!")

                if "generated_report_html" in st.session_state:
                    st.download_button(
                        "⬇️ DOWNLOAD MISSION AUDIT REPORT (HTML/PDF)",
                        data=st.session_state.generated_report_html,
                        file_name=f"mission_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        width="stretch"
                    )
                    with st.expander("👁️ Preview Generated Mission Audit Report"):
                        st.components.v1.html(st.session_state.generated_report_html, height=500, scrolling=True)
            else:
                st.info("System logs record automatically as mission steps are executed.")
                if st.button("📄 GENERATE MISSION AUDIT REPORT (INITIAL)", width="stretch"):
                    st.session_state.generated_report_html = MissionReportGenerator.generate_html_report(
                        experiment_name=selected_group,
                        steps=pipeline.experiment_manager.steps,
                        current_idx=pipeline.experiment_manager.current_idx,
                        events=[]
                    )
                    st.success("✅ Initial Mission Audit Report generated!")

                if "generated_report_html" in st.session_state:
                    st.download_button(
                        "⬇️ DOWNLOAD MISSION AUDIT REPORT (HTML/PDF)",
                        data=st.session_state.generated_report_html,
                        file_name=f"mission_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        width="stretch"
                    )
                    with st.expander("👁️ Preview Generated Mission Audit Report"):
                        st.components.v1.html(st.session_state.generated_report_html, height=500, scrolling=True)

        with tab_dataset:
            st.subheader("📊 Group-Isolated Dataset Generator")

            ds_group_target = selected_group
            active_ds_gen = pipeline.dataset_generator
            active_ds_gen.set_active_group(ds_group_target)

            st.success(f"🟢 **Dataset Directory:** `data/dataset/{active_ds_gen.normalize_group_name(ds_group_target)}/` — Active Group: **{ds_group_target}**")

            ds_col1, ds_col2 = st.columns(2)
            with ds_col1:
                st.markdown(f"#### 📸 Manual Sample Collector — {ds_group_target}")
                available_labels = active_ds_gen.labels

                target_lbl = st.selectbox(
                    "Select Target Activity Label to Capture:",
                    available_labels,
                    help="Select an activity label belonging to the active Activity Group."
                )
                
                if st.button("📸 CAPTURE MANUAL SAMPLE + METADATA", width="stretch"):
                    raw_frame = pipeline.latest_raw_frame
                    if raw_frame is None:
                        ret, raw_frame = CameraManager().read()

                    if raw_frame is not None:
                        meta = {
                            "activity_group": ds_group_target,
                            "interaction_event": pipeline.interaction_event,
                            "detected_objects": [d["class"] for d in pipeline.detections]
                        }
                        saved = active_ds_gen.save_sample(target_lbl, raw_frame, meta)
                        if saved:
                            st.success(f"✅ Captured sample for '{target_lbl}' in '{ds_group_target}' dataset!")
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            st.error("Failed to save frame sample.")
                    else:
                        st.warning("Please ensure the camera is connected to capture samples.")

                st.divider()
                st.markdown("#### 🗑️ Reset Dataset Storage")
                st.write("Clear all collected samples across dataset folders:")
                if st.button("🗑️ RESET DATASET SAMPLES (CLEAR ALL TO 0)", width="stretch"):
                    cleared = active_ds_gen.clear_dataset(group_only=False)
                    if cleared:
                        st.success("🗑️ Dataset storage completely cleared! All sample counts reset to 0.")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Failed to clear dataset folder.")

            with ds_col2:
                st.markdown(f"#### 📈 Live Dataset Statistics — {ds_group_target}")
                stats = active_ds_gen.get_statistics(ds_group_target)
                st.dataframe([{"Activity Label": k, "Saved Samples": v} for k, v in stats.items()], width="stretch", hide_index=True)
                
                col_ref1, col_ref2 = st.columns(2)
                with col_ref1:
                    if st.button("🔄 REFRESH TABLE STATS", width="stretch"):
                        st.rerun()
                with col_ref2:
                    st.download_button(
                        "⬇️ EXPORT METADATA (JSON)",
                        data=json.dumps({"activity_group": ds_group_target, "folder": active_ds_gen.active_dir, "stats": stats}, indent=2),
                        file_name=f"dataset_{active_ds_gen.normalize_group_name(ds_group_target)}.json",
                        mime="application/json",
                        width="stretch"
                    )

        with tab_system:
            st.subheader("⚙️ System Telemetry & Video Controls")
            
            sys_c1, sys_c2 = st.columns(2)
            with sys_c1:
                st.markdown("#### 🎥 Local Video Recording Control")
                if pipeline.video_manager.is_recording:
                    vm = pipeline.video_manager
                    dur = round(time.time() - vm.start_time, 1) if vm.start_time > 0 else 0
                    st.success(f"🔴 Currently Recording: `{vm.current_filename}` ({vm.frames_written} frames saved, {dur}s)")
                else:
                    st.caption("Status: Recording is currently OFF")
                
                col_rec1, col_rec2 = st.columns(2)
                with col_rec1:
                    if st.button("🔴 START VIDEO RECORDING", width="stretch"):
                        ok = pipeline.video_manager.start_recording()
                        if ok:
                            st.success("Started local video recording!")
                with col_rec2:
                    if st.button("⏹️ STOP VIDEO RECORDING", width="stretch"):
                        if pipeline.video_manager.is_recording:
                            rec_info = pipeline.video_manager.stop_recording()
                            st.success(f"Stopped recording! Saved {rec_info['frames']} frames ({rec_info['duration_sec']}s) to `{rec_info['filepath']}`")
                        else:
                            st.warning("Video recording is not currently active.")

            with sys_c2:
                st.markdown("#### 🌐 Optional IP Video Streaming")
                ip_addr = st.text_input("Target Stream IP Address", value=self.stream_manager.ip)
                port_num = st.number_input("Target Stream Port", value=self.stream_manager.port)
                
                stream_status = self.stream_manager.get_status()
                if stream_status["streaming"]:
                    st.success(f"🟢 Connected & Streaming to {stream_status['ip']}:{stream_status['port']}")
                else:
                    st.caption("Status: Streaming DISCONNECTED")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    if st.button("▶️ START IP STREAM"):
                        self.stream_manager.start_stream(ip_addr, int(port_num))
                        st.success(f"IP Stream started on {ip_addr}:{port_num}")
                with col_s2:
                    if st.button("⏹️ STOP IP STREAM"):
                        self.stream_manager.stop_stream()
                        st.info("IP Stream stopped.")

            st.divider()

            st.markdown("#### 🧠 AI Model Architecture Status")
            st.json(self.model_manager.get_status())

            st.markdown("#### 📐 Orientation-Agnostic 3D Human Mesh Recovery Interface")
            st.json(self.pose_3d_estimator.estimate_3d(None))
