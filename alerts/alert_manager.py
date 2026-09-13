import time
import threading
import sys
import subprocess
from typing import Optional, Dict, Any, List

_thread_local = threading.local()


def _get_thread_speaker():
    """Returns a platform-specific speech backend."""

    # macOS
    if sys.platform == "darwin":
        return "macos"

    # Windows
    if sys.platform == "win32":
        if not hasattr(_thread_local, "speaker") or _thread_local.speaker is None:
            try:
                import pythoncom
                pythoncom.CoInitialize()

                import win32com.client

                sp = win32com.client.Dispatch("SAPI.SpVoice")
                sp.Rate = 1
                sp.Volume = 100

                _thread_local.speaker = sp

            except Exception as e:
                print(f"[AlertManager] Windows SAPI init note: {e}")
                _thread_local.speaker = None

        return getattr(_thread_local, "speaker", None)

    return None


class AlertManager:

    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown = cooldown_seconds
        self.last_alert_time = 0.0
        self.last_alert_message = ""
        self.event_counter = 0
        self.latest_voice_event: Optional[Dict[str, Any]] = None

    def purge_speech(self):
        """Stop any currently playing speech."""

        try:
            sp = _get_thread_speaker()

            if sp == "macos":
                subprocess.run(
                    ["killall", "say"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            elif sp is not None:
                # Windows SAPI purge
                sp.Speak("", 2)

        except Exception as e:
            print(f"[AlertManager] Purge speech error: {e}")

    def speak_instant(self, text: str):
        """Speak an alert without blocking the video-processing thread."""

        if not text:
            return

        try:
            sp = _get_thread_speaker()

            # =========================
            # macOS
            # =========================
            if sp == "macos":

                # Stop previous speech
                subprocess.run(
                    ["killall", "say"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Start new speech asynchronously
                process = subprocess.Popen(
                    ["say", "-r", "190", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                print(f"[AlertManager] macOS voice alert started: {text}")

            # =========================
            # Windows
            # =========================
            elif sp is not None:

                # Async + purge previous speech
                sp.Speak(text, 3)

                print(f"[AlertManager] Windows voice alert started: {text}")

            else:
                print(
                    f"[AlertManager] No supported speech backend "
                    f"for platform: {sys.platform}"
                )

        except Exception as e:
            print(f"[AlertManager] Speech error: {e}")

    def trigger_wrong_step_alert(
        self,
        expected: str,
        detected: str
    ) -> Optional[str]:

        now = time.time()

        if now - self.last_alert_time >= self.cooldown:

            msg = (
                f"Sequence Alert. "
                f"Detected {detected}. "
                f"Expected {expected}."
            )

            self.last_alert_time = now
            self.last_alert_message = msg
            self.event_counter += 1

            self.latest_voice_event = {
                "id": self.event_counter,
                "type": "voice_alert",
                "message": msg,
                "severity": "warning",
                "timestamp": now
            }

            self.speak_instant(msg)

            return msg

        return None

    def trigger_missing_object_alert(
        self,
        expected_obj: str,
        detected_classes: Optional[List[str]] = None
    ) -> Optional[str]:

        now = time.time()

        if now - self.last_alert_time >= self.cooldown:

            if detected_classes and len(detected_classes) > 0:

                det_str = ", ".join(detected_classes)

                msg = (
                    f"Object Alert. "
                    f"Expected {expected_obj}, "
                    f"but detected {det_str}."
                )

            else:

                msg = (
                    f"Attention. "
                    f"Required object {expected_obj} is missing."
                )

            self.last_alert_time = now
            self.last_alert_message = msg
            self.event_counter += 1

            self.latest_voice_event = {
                "id": self.event_counter,
                "type": "voice_alert",
                "message": msg,
                "severity": "warning",
                "timestamp": now
            }

            self.speak_instant(msg)

            return msg

        return None

    def trigger_correct_step_alert(
        self,
        step_name: str
    ) -> Optional[str]:

        now = time.time()

        if now - self.last_alert_time >= 1.2:

            msg = f"Step verified: {step_name}."

            self.last_alert_time = now
            self.last_alert_message = msg
            self.event_counter += 1

            self.latest_voice_event = {
                "id": self.event_counter,
                "type": "voice_alert",
                "message": msg,
                "severity": "info",
                "timestamp": now
            }

            self.speak_instant(msg)

            return msg

        return None

    def trigger_completion_alert(self) -> Optional[str]:

        msg = (
            "Mission experiment sequence "
            "successfully completed."
        )

        self.last_alert_message = msg
        self.event_counter += 1

        self.latest_voice_event = {
            "id": self.event_counter,
            "type": "voice_alert",
            "message": msg,
            "severity": "success",
            "timestamp": time.time()
        }

        self.speak_instant(msg)

        return msg

    def get_latest_voice_event(
        self
    ) -> Optional[Dict[str, Any]]:

        return self.latest_voice_event